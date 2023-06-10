# NLI Task

> NLU 2023 Course Project

- Data should be `data/train.jsonl` and `data/test.jsonl`

  - Each line should be `{text_a:"sencence A.", text_b:"sencence B.", label:"entailment"}` and test dataset has no label

## Get Started

1. Run `check_length.py` and determine the proper max_length for tokenizer

2. Run `preprocess.py` to tokenize the dataset

3. Run `nli.py`