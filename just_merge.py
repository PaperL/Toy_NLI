import json

def merge_datasets(file1, file2, output_file):
    # Load data from the first file
    with open(file1, 'r') as f1:
        data1 = [json.loads(line) for line in f1]

    # Load data from the second file
    with open(file2, 'r') as f2:
        data2 = [json.loads(line) for line in f2]

    # Merge the datasets
    merged_data = data1 + data2

    # Save the merged dataset to a new file
    with open(output_file, 'w') as output_f:
        for line in merged_data:
            output_f.write(json.dumps(line) + '\n')

    print(f"Merged dataset saved to {output_file}")


# Example usage
merge_datasets('data/train.jsonl', 'data/augment.jsonl', 'data/train_simple_aug.jsonl')
