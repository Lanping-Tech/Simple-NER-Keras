import os
import numpy as np
from tqdm import tqdm

UNK, PAD = '<UNK>', '<PAD>'

def build_vocab(vocab_path):
    vocab_dict = {}
    for file_path in vocab_path:
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                word = line.strip()
                word = word.split(' ')[0]
                if word:
                    vocab_dict[word] = vocab_dict.get(word, 0) + 1
    
    vocab_list = sorted([_ for _ in vocab_dict.items()], key=lambda x: x[1], reverse=True) # if _[1] >= 1
    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})
    return vocab_dict

def label_map(label):
    if label.endswith('O'): 
        return 0
    elif label.endswith('PER'):
        return 1
    elif label.endswith('LOC'):
        return 2
    elif label.endswith('ORG'):
        return 3
    elif label.endswith('MISC'):
        return 4
    else:
        print("Error in label!!!" + str(label))

def load_data(data_path, vocab_dict, max_seq_len):
    with open(data_path, 'r', encoding='UTF-8') as f:
        lines = [line.strip() for line in f]

    sentence = []
    sentence_tag = []
    sentences = []
    sentences_tag = []

    for line in tqdm(lines):
        if line:
            word = line.split(' ')[0]
            tag = line.split(' ')[-1]
            sentence.append(word)
            sentence_tag.append(tag)
        else:
            if sentence:
                seq_len = len(sentence)
                if seq_len < max_seq_len:
                    sentence.extend([PAD] * (max_seq_len - seq_len))
                    sentence_tag.extend(['O'] * (max_seq_len - seq_len))
                else:
                    sentence = sentence[:max_seq_len]
                    sentence_tag = sentence_tag[:max_seq_len]
                
                for idx, word in enumerate(sentence):
                    if word not in vocab_dict:
                        sentence[idx] = UNK
                        sentence_tag[idx] = 'O'
                sentence = [vocab_dict[word] for word in sentence]
                sentence_tag = [label_map(label) for label in sentence_tag]
                sentences.append(sentence)
                sentences_tag.append(sentence_tag)
                sentence, sentence_tag = [], []

            else:
                continue


    
    return np.array(sentences), np.array(sentences_tag)