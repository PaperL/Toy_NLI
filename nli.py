import torch
import csv
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from py_console import console

# Define the labels and their corresponding descriptions
LABELS = {
    0: 'entailment',
    1: 'contradiction',
    2: 'neutral'
}
DEBUG = False

# Define the dataset class


class NLIDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('./pretrained')
        console.info(f'Reading data from {file_path}')
        with open(file_path, 'r') as file:
            _cnt = 0
            for line in tqdm(file):
                if DEBUG and _cnt > 100:
                    break
                _cnt += 1
                example = eval(line)
                input_ids = torch.tensor(example['input_ids'])
                attention_mask = torch.tensor(example['attention_mask'])
                token_type_ids = torch.tensor(example['token_type_ids'])
                label = example.get('label', len(LABELS)+1)
                self.data.append(
                    (input_ids, attention_mask, token_type_ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Define the model class


class NLIModel(torch.nn.Module):
    def __init__(self):
        super(NLIModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            './pretrained', num_labels=len(LABELS))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        probabilities = self.softmax(logits)
        return probabilities

# Function to convert integer labels to their corresponding descriptions


def label_to_description(label):
    return LABELS[label]

# Function to evaluate the model and write the results to a CSV file


def evaluate(model, dataloader, file_path):
    console.info('Evaluating')
    model.eval()
    predictions = []
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'label'])
        id = 0
        for i, (input_ids, attention_mask, token_type_ids, _) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            probabilities = model(input_ids, attention_mask, token_type_ids)
            _, predicted_labels = torch.max(probabilities, dim=1)
            predicted_labels = predicted_labels.tolist()
            for label in predicted_labels:
                predictions.append(label_to_description(label))
                writer.writerow([id, label_to_description(label)])
                id += 1
    console.success("Evaluation results saved to `evaluation_results.csv`")
    return predictions


# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    console.success('Using GPU')
else:
    console.warning('Using CPU')

# Set the paths to the training and test data files
train_file_path = 'data/train_simple_aug_token.jsonl'
test_file_path = 'data/test_token.jsonl'
console.info(f'Training data: {train_file_path}')
console.info(f'Test data: {test_file_path}')


# Set the hyperparameters
batch_size = 32
learning_rate = 2e-5
num_epochs = 100

# Create the datasets and dataloaders
train_dataset = NLIDataset(train_file_path)
test_dataset = NLIDataset(test_file_path)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the NLIModel
model = NLIModel().to(device)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
console.success('Begin training')
for epoch in tqdm(range(num_epochs), desc='Epoch'):
    model.train()
    running_loss = 0.0
    batch_progress_bar = tqdm(enumerate(train_dataloader),
                              total=len(train_dataloader), desc='Batch', leave=False)
    for i, (input_ids, attention_mask, token_type_ids, labels) in batch_progress_bar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        probabilities = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(probabilities, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_progress_bar.set_postfix({'loss': loss.item()})

    average_loss = running_loss / len(train_dataloader)
    # Save model
    torch.save(model.state_dict(), f"model/0001_{epoch}.pth")
    console.info(
        f"Epoch {epoch}/{num_epochs} - Average Loss: {average_loss:.4f}")
    # Evaluation
    predictions = evaluate(model, test_dataloader, f'output/0001_{epoch}.csv')

