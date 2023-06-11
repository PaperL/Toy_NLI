import json
from multiprocessing import Pool
import nlpaug.augmenter.word as naw
import spacy
from tqdm import tqdm

DEBUG = False

nlp = spacy.load('en_core_web_sm')

# Step 1: Read data from train.jsonl
input_file = 'data/train.jsonl'
output_file = 'data/augment.jsonl'

data = []
with open(input_file, 'r') as file:
    for line in tqdm(file):
        data.append(json.loads(line.strip()))
        if DEBUG and len(data) > 10:
            break

# Step 2: Extract adjectives and verbs from text_a and text_b


def extract_tokens(sentence):
    tokens = []
    doc = nlp(sentence)
    for token in doc:
        if token.tag_.startswith('JJ') or token.tag_.startswith('VB'):
            tokens.append(token)
    return tokens


syn_augmenter = naw.SynonymAug()
ant_augmenter = naw.AntonymAug()

# Step 3 and 4: Replace adjectives and verbs and generate new data
# for example in tqdm(data, desc='Process'):

def augment_work(example):
    augmented_data = []
    text2 = [example['text_a'], example['text_b']]
    label = example['label']
    tokens2 = [extract_tokens(text) for text in text2]
    syn_augmented2 = [text for text in text2]
    ant_augmented2 = [text for text in text2]
    ant_cnt = 0

    for i in range(2):
        for token in tokens2[i]:
            syn_augmented_word = syn_augmenter.augment(
                token.text, num_thread=1, n=1)[0]
            ant_augmented_word = ant_augmenter.augment(
                token.text, num_thread=1, n=1)[0]
            if token.text != ant_augmented_word:
                ant_cnt += 1
            for syn_augmented_token in nlp(syn_augmented_word):
                if token.tag_ == syn_augmented_token.tag_:
                    syn_augmented2[i] = syn_augmented2[i].replace(
                        token.text, syn_augmented_word)
                    if DEBUG:
                        print('SYN ', token.text, '<', syn_augmented_word, '>')
            for ant_augmented_token in nlp(ant_augmented_word):
                if token.tag_ == ant_augmented_token.tag_:
                    if ant_cnt % 2 == 0 and label != 'contradiction':
                        ant_augmented2[i] = ant_augmented2[i].replace(
                            token.text, ant_augmented_word)
                    if DEBUG:
                        print('ANT ', token.text, '<', ant_augmented_word, '>')

    if syn_augmented2[0] != text2[0] or syn_augmented2[1] != text2[1]:
        augmented_example = {
            'text_a': syn_augmented2[0],
            'text_b': syn_augmented2[1],
            'label': label
        }
        augmented_data.append(augmented_example)
        if DEBUG:
            for jj in range(2):
                if syn_augmented2[jj] != text2[jj]:
                    print('SYN')
                    print(syn_augmented2[jj], text2[jj], sep='\n')

    if ant_augmented2[0] != text2[0] or ant_augmented2[1] != text2[1]:
        reversed_label = label
        if label == 'entailment':
            reversed_label = 'contradiction'
        augmented_example = {
            'text_a': ant_augmented2[0],
            'text_b': ant_augmented2[1],
            'label': reversed_label
        }
        augmented_data.append(augmented_example)
        if DEBUG:
            for jj in range(2):
                if ant_augmented2[jj] != text2[jj]:
                    print('ANT')
                    print(ant_augmented2[jj], text2[jj], sep='\n')
    
    return augmented_data


all_augmented_data = []
with Pool() as pool:
    for result in tqdm(pool.imap_unordered(augment_work, data), total=len(data)):
        all_augmented_data.extend(result)

# Step 5: Save augmented data in aug.jsonl
with open(output_file, 'w') as file:
    for example in all_augmented_data:
        file.write(json.dumps(example) + '\n')
