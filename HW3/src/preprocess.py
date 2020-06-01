## Script for Preprocess the data

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchtext
import re
import random

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, KeyedVectors
from collections import Counter

# Set random seed
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Set path
PRETRRINED_path = os.path.join(os.getcwd(), 'data', 'all.review.vec.txt')
TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'train')
TEST_PATH = os.path.join(os.getcwd(), 'data', 'test')

if __name__ == '__main__':
    # Check time
    start_time = time.perf_counter()
    
    # Read Pretrained Data
    # Use vector format instead
    words_set = set()
    word_embedding_pre = {}
    
    idx=0
    with open(PRETRRINED_path) as f:
        for line in f:
            line_list = line.split()
            if len(line_list) == 101:
                word = line_list[0]
                words_set.add(word)
                word_embedding_pre[word] = np.array(line_list[1:], dtype=float)
    
    
    # Setup data_loader with preprocess
    def data_loader(FILE_PATH):
        Data = []
        Data_df = pd.DataFrame(columns=['label', 'text'])
        for sense in os.listdir(FILE_PATH):
            sense_folder_path = os.path.join(FILE_PATH, sense)
            for file in os.listdir(sense_folder_path):
                sense_file_path = os.path.join(sense_folder_path, file)
                with open(sense_file_path) as f:
                    for line in f:
                        # line_list = line.split()
    
                        # Add preprocess & clean the sentence
                        # Save data into unit of sentence or word
                        content_text = re.sub(r'\([^)]*\)', '', line)
                        sent_text = sent_tokenize(content_text)
    
                        normalized_text = []
                        for string in sent_text:
                            tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
                            normalized_text.append(tokens)
    
                        result_content = ''.join(normalized_text)
                        result_sentence = [word_tokenize(sentence) for sentence in normalized_text]
                        result = [word for sentence in result_sentence for word in sentence]
    
                        if sense == 'positive':
                            Data.append({'label': 'pos', 'text': result})
                            Data_df = Data_df.append([{'label': 'pos', 'text': result_content}], ignore_index=True,
                                                     sort=False)
                        elif sense == 'negative':
                            Data.append({'label': 'neg', 'text': result})
                            Data_df = Data_df.append([{'label': 'neg', 'text': result_content}], ignore_index=True,
                                                     sort=False)
        return Data, Data_df
    
    
    # Load data
    Train_data, Train_data_df = data_loader(TRAIN_PATH)
    Test_data, Test_data_df = data_loader(TEST_PATH)
    
    # Save as a csv format to data folder
    Train_data_df.to_csv('data/train_df.csv', index=False)
    Test_data_df.to_csv('data/test_df.csv', index=False)
    
    # Define train_set / val_set / test_set
    train_set_df, validation_set_df = train_test_split(Train_data_df, test_size=0.2, random_state=SEED)
    test_set_df = Test_data_df.copy()
    
    # Save as a csv format
    train_set_df.to_csv('train_set_df.csv', index=False)
    validation_set_df.to_csv('validation_set_df.csv', index=False)
    test_set_df.to_csv('test_set_df.csv', index=False)
    
    print('----- Save data: train / valid / test  -----')
    
    ## Train data exploration
    def Text_to_words(Train_data):
        Train_total_words = []
    
        for sentence in Train_data:
            for word in sentence['text']:
                Train_total_words.append(word)
        return np.array(Train_total_words)
    
    Train_total_words = Text_to_words(Train_data)
    
    ##Setup model dimensions
    #a.	Word embedding dimension: 100
    #b.	Word Index: keep the most frequent 10k words
    
    Train_word_count = Counter(Train_total_words).most_common(10000)
    
    word_to_idx = {}
    word_to_idx['<unk>'] = 0
    word_to_idx['<pad>'] = 1
    
    idx = 2
    for word, count in Train_word_count:
        word_to_idx[word] = idx
        idx += 1
    
    # Create word2vector model
    Train_data_sentence = [T['text'] for T in Train_data]
    W2V_model = Word2Vec(sentences=Train_data_sentence, size=100, window=5, min_count=3, sg=0)
    
    W2V_model.wv.save_word2vec_format('./w2v_model')
    loaded_model = KeyedVectors.load_word2vec_format("w2v_model")
    W2V_model = loaded_model
    
    word_embedding_W2V = dict()
    word_embedding_W2V['<unk>'] = (np.random.random_sample(100)-0.5)*0.1 #random between -0.05~0.05
    word_embedding_W2V['<pad>'] = np.zeros(100)
    
    for idx, word in enumerate(word_to_idx):
        try:
            word_embedding_W2V[word] = W2V_model[word]
        except KeyError:
            continue
            #print(word, 'is not trained ')
    
    print('----- Save Word2Vector embedding model -----')
    
    print('----- Total Time: {} (s)      -----'.format(time.perf_counter() - start_time))