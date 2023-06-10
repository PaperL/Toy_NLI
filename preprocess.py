import json
import transformers
from transformers import BertTokenizer
from tqdm import tqdm 

transformers.logging.set_verbosity_error() # Close the annoying warning

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_dataset(input_file, output_file, is_train=True):
    with open(input_file, 'r') as file:
        data = [json.loads(line) for line in file]

    preprocessed_data = []
    for example in tqdm(data):
        text_a = example['text_a']
        text_b = example['text_b']
        label = example.get('label', None)

        # Tokenize the input text
        encoded_inputs = tokenizer.encode_plus(
            text_a, text_b,
            add_special_tokens=True,
            padding='max_length',
            max_length=128,
            truncation=True)

        preprocessed_example = {
            'input_ids': encoded_inputs['input_ids'],
            'attention_mask': encoded_inputs['attention_mask'],
            'token_type_ids': encoded_inputs['token_type_ids'],
        }

        # Convert label to integer
        if is_train:
            if label == 'entailment':
                label_id = 0
            elif label == 'contradiction':
                label_id = 1
            elif label == 'neutral':
                label_id = 2
            else:
                raise ValueError(f"Unknown label: {label}")
            preprocessed_example['label'] = label_id

        preprocessed_data.append(preprocessed_example)

    with open(output_file, 'w') as file:
        for example in preprocessed_data:
            file.write(json.dumps(example) + '\n')


# Preprocess train dataset
preprocess_dataset('data/train.jsonl', 'data/train_token.jsonl')

# Preprocess test dataset
preprocess_dataset('data/test.jsonl', 'data/test_token.jsonl', is_train=False)
