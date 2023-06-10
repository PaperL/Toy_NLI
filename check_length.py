import jsonlines
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# Load the dataset from train.jsonl
data = []
with jsonlines.open('data/train.jsonl') as reader:
    for obj in reader:
        data.append(obj)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Calculate sequence lengths
sequence_lengths = []

for example in tqdm(data):
    text_a = example['text_a']
    text_b = example['text_b']

    encoded = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        return_tensors='np'  # Use NumPy tensors
    )

    sequence_length = encoded['input_ids'].shape[1]
    sequence_lengths.append(sequence_length)

bins = [x*8 for x in range(0, 17)]
bins.append(max(sequence_lengths))

# Plot the distribution of sequence lengths
plt.hist(sequence_lengths, bins=bins, edgecolor='black')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Distribution of Sequence Lengths')
print('f1')

# Add count labels to each column
bin_counts, _, _ = plt.hist(sequence_lengths, bins=bins)
for i, count in enumerate(bin_counts):
    plt.text((bins[i] + bins[i+1]) / 2, count, str(int(count)), ha='center')
print('f2')

# Save the plot as an image
plt.tight_layout()
plt.savefig('sequence_length_distribution.png')
plt.close()
