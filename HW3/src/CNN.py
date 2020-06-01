## Script for Preprocess the data

import os
import time
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchtext
import re
import random

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, KeyedVectors
from collections import Counter

# Module for data processing and model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.vocab import Vectors, Vocab
from torchtext.vocab import GloVe
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator


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

# argparse for parameters
def parse_args():
    parser = argparse.ArgumentParser(description='Python module for Neural Network model for sensitive analysis')
    parser.add_argument('--Embedd', type=str, required=True, help='Embedding methods: W2V | Pre | G6B')
    parser.add_argument('--Epochs', type=int, required=False, default=10, help='Epoch for model train (default: 10)')

    return parser.parse_args()

# Setup hyper-parameters
NUM_WORDS = 1000
NUM_DIM = 100
BATCH_SIZE = 64
NUM_CLASS = 2

# Define model
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


def train(model, optimizer, train_iter):
    model.train()
    corrects, total_loss = 0, 0
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(train_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    with torch.no_grad():
        for batch in val_iter:
            x, y = batch.text.to(device), batch.label.to(device)
            y.data.sub_(1)
            logit = model(x)
            loss = F.cross_entropy(logit, y, reduction='sum')
            total_loss += loss.item()
            corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


if __name__ == '__main__':
    # Check time
    start_time = time.perf_counter()
    params = parse_args()

    # parameter initialization -----------------------------------------------------------------------------------------
    Embedding_set = ['W2V', 'Pre', 'G6B']
    if params.Embedd not in Embedding_set:
        raise ValueError('Not valid method')
    Embedding = params.Embedd

    EPOCHS = params.Epochs

    Model = 'CNN'

    # Define embedding vector
    w2v_vectors = Vectors(name='w2v_model')
    pre_vectors = Vectors(PRETRRINED_path)

    # Define Torchtext structure
    LABEL = torchtext.data.Field(sequential=False, use_vocab=True,
                                batch_first=False, is_target=True)

    TEXT = torchtext.data.Field(sequential=True, use_vocab=True,
                                tokenize=str.split, lower=True,
                                batch_first=True, fix_length=NUM_WORDS)

    # Load data
    train_data, valid_data, test_data = TabularDataset.splits(
        path='.', train='train_set_df.csv', validation='validation_set_df.csv', test='test_set_df.csv',
        format='csv', fields=[('label', LABEL), ('text', TEXT)], skip_header=True)

    # Build vocab with embedding vector
    #TEXT.build_vocab(train_data, min_freq=3, max_size=10000)
    if Embedding == 'W2V':
        TEXT.build_vocab(train_data, vectors=w2v_vectors , min_freq=3, max_size=10000)
    elif Embedding == 'Pre':
        TEXT.build_vocab(train_data, vectors=pre_vectors , min_freq=3, max_size=10000)
    elif Embedding == 'G6B':
        TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100) , min_freq=3, max_size=10000)
    else:
        raise(ValueError('Not Valid method'))
    LABEL.build_vocab(train_data)

    VOCAB_SIZE = len(TEXT.vocab)

    # Define data bucket and iterator
    train_loader = Iterator(dataset=train_data, batch_size = BATCH_SIZE, device = device)
    valid_loader = Iterator(dataset=valid_data, batch_size = BATCH_SIZE, device = device)
    test_loader = Iterator(dataset=test_data, batch_size = BATCH_SIZE, device = device)

    train_iter, valid_iter, test_iter = BucketIterator.splits(
                                                (train_data, valid_data, test_data),
                                                batch_size = BATCH_SIZE,
                                                sort_key=lambda x: len(x.text),
                                                sort_within_batch = False,
                                                shuffle=True, repeat=False, #sort=False,
                                                device = device)

    print('The number of mini-batch in train_data : {}'.format(len(train_iter)))
    print('The number of mini-batch in validation_data : {}'.format(len(valid_iter)))
    print('The number of mini-batch in test_data : {}'.format(len(test_iter)))

    # Define model
    INPUT_DIM = VOCAB_SIZE
    EMBEDDING_DIM = NUM_DIM
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 2
    DROPOUT = 0.5

    model = CNN(INPUT_DIM,
                EMBEDDING_DIM,
                N_FILTERS,
                FILTER_SIZES,
                OUTPUT_DIM,
                DROPOUT)

    model.to(device)

    # Setup embedding parameters
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Train and Evaluate Model
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = None
    train_out = []
    valid_out = []
    test__out = []
    
    for e in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train(model, optimizer, train_iter)
        valid_loss, valid_accuracy = evaluate(model, valid_iter)
        test__loss, test__accuracy = evaluate(model, test_iter)
    
        train_out.append([train_loss, train_accuracy])
        valid_out.append([valid_loss, valid_accuracy])
        test__out.append([test__loss, test__accuracy])
    
        #    print("[Epoch: %d] train loss : %5.2f | train accuracy : %5.2f" % (e, train_loss, train_accuracy))
        print("[Epoch: %d] valid loss : %5.2f | valid accuracy : %5.2f" % (e, valid_loss, valid_accuracy))
        print("[Epoch: %d] test  loss : %5.2f | test  accuracy : %5.2f" % (e, test__loss, test__accuracy))
    
        if not best_val_loss or valid_loss < best_val_loss:
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), './snapshot/CNN_classification.pt')
            best_val_loss = valid_loss
    
    model.load_state_dict(torch.load('./snapshot/CNN_classification.pt'))
    test_loss, test_acc = evaluate(model, test_iter)
    print('Test Loss: %5.2f | Test Accuracy: %5.2f' % (test_loss, test_acc))
    
    ## Save Figure
    plt.figure()
    plt.plot(np.array(train_out)[:,0])
    plt.plot(np.array(valid_out)[:,0])
    plt.plot(np.array(test__out)[:,0])
    plt.legend(['train', 'valid', 'test'])
    plt.title('loss'+'_'+str(Embedding)+'_'+str(Model))
    plt.savefig('loss'+'_'+str(Embedding)+'_'+str(Model)+'.png')
    #plt.show()
    
    plt.figure()
    plt.plot(np.array(train_out)[:,1])
    plt.plot(np.array(valid_out)[:,1])
    plt.plot(np.array(test__out)[:,1])
    plt.legend(['train', 'valid', 'test'])
    plt.title('accuracy'+'_'+str(Embedding)+'_'+str(Model))
    plt.savefig('accuracy'+'_'+str(Embedding)+'_'+str(Model)+'.png')
    #plt.show()
    
    print('Save Output: {}_{}'.format(Embedding, Model))
    print('Total Time: {} (s)'.format(time.perf_counter() - start_time))