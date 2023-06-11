import json
import random

def replace_lines(train_file, augment_file, output_file):
    # Load the train.jsonl file
    with open(train_file, 'r') as train_f:
        train_data = [json.loads(line) for line in train_f]

    # Load the augment.jsonl file
    with open(augment_file, 'r') as augment_f:
        augment_data = [json.loads(line) for line in augment_f]

    # Generate a random sequence of indices to replace in train_data
    num_lines_to_replace = min(len(train_data), len(augment_data))
    replace_indices = random.sample(range(len(train_data)), num_lines_to_replace)

    # Replace lines in train_data with lines from augment_data
    for i, index in enumerate(replace_indices):
        train_data[index] = augment_data[i]

    # Save the augmented train_data to a new file
    with open(output_file, 'w') as output_f:
        for line in train_data:
            output_f.write(json.dumps(line) + '\n')

    print(f"Augmented dataset saved to {output_file}")


# Example usage
replace_lines('data/train.jsonl', 'data/augment.jsonl', 'data/train_augmented.jsonl')
