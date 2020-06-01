import os
import sys
import json
import re
import math
import numpy as np
import pandas as pd
import pickle

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from gensim.models import Word2Vec, KeyedVectors

np.random.seed(0)

#Set File Path
stopword_path = 'data/stopword.list'       
train_path = 'data/yelp_reviews_train.json'
test_path = 'data/yelp_reviews_test.json'  
dev_path = 'data/yelp_reviews_dev.json'  

#Set parameters
feature_num = 2000
class_num = 5


#Tokenize function
def tokenize(text, stop_word):
    text_tokens = []
    text = text.lower()
    text = re.sub('[^A-Za-z0-9\s]+', '', text).split()
    for token in text:
        if token not in stop_word and not any(t.isdigit() for t in token):
            text_tokens.append(token.strip())
    return text_tokens

def review_to_csr_matrix(text_token_list, feature_list, feature_num):
    review_row = []
    review_col = []
    review_data = []
    
    line_num = 0
    for token_list in text_token_list:
        feature_count = np.array([token_list[f] for f in feature_list])
        col_idx = np.array(range(feature_num))
        col = col_idx[feature_count!=0]
        row = [line_num]*len(col)
        data = feature_count[feature_count!=0]

        review_row.append(row)
        review_col.append(col)
        review_data.append(data)

        line_num += 1   

    #Flatten the list of list
    review_row = [item for sub in review_row for item in sub]
    review_col = [item for sub in review_col for item in sub]
    review_data = [item for sub in review_data for item in sub]
     
    review_matrix = csr_matrix((review_data, (review_row, review_col)), shape=(len(text_token_list), feature_num))
    
    return review_matrix

def stars_to_csr_matrix(stars_list, class_num):
    stars_row = range(len(stars_list))
    stars_col = stars_list
    stars_data = [1] * len(stars_list)
    stars_matrix = csr_matrix((stars_data, (stars_row, stars_col)), shape=(len(stars_list), class_num+1)) #Add 0 for non-label data
    
    return stars_matrix


if __name__ == '__main__':
    method = str(sys.argv[1]) #[CTF or DF or BAL]
    method_set = ['CTF', 'DF', 'BAL']
    if method not in method_set:
        raise(ValueError('Not Valid method'))
    print('{} Start'.format(method))
    
    with open(stopword_path) as f:
        stopword_lines = f.readlines()
    stop_word = set(line.strip() for line in stopword_lines)
    
    #Read Train
    train_review_list = []
    train_stars_list = []
    cnt = 0
    with open(train_path) as f:
        train_line = f.readline()
        while train_line != '':
            review = json.loads(train_line)
            review_id = review['review_id']
            review_text = review['text']
            review_stars = int(review['stars'])

            review_token_list = tokenize(review_text, stop_word)
            train_review_list.append(review_token_list)
            train_stars_list.append(review_stars)

            train_line = f.readline()
            cnt += 1

    print("  Train data size: {}".format(cnt))
    
    #Read Dev
    dev_review_list = []
    dev_stars_list = []
    cnt = 0
    with open(dev_path) as f:
        dev_line = f.readline()
        while dev_line != '':
            review = json.loads(dev_line)
            review_id = review['review_id']
            review_text = review['text']
            #review_stars = int(review['stars'])

            review_token_list = tokenize(review_text, stop_word)
            dev_review_list.append(review_token_list)
            dev_stars_list.append(0)

            dev_line = f.readline()
            cnt += 1

    print("  Dev data size: {}".format(cnt))    
    
    #Read Test
    test_review_list = []
    test_stars_list = []
    cnt = 0
    with open(test_path) as f:
        test_line = f.readline()
        while test_line != '':
            review = json.loads(test_line)
            review_id = review['review_id']
            review_text = review['text']
            #review_stars = int(review['stars'])

            review_token_list = tokenize(review_text, stop_word)
            test_review_list.append(review_token_list)
            test_stars_list.append(0)

            test_line = f.readline()
            cnt += 1

    print("  Test data size: {}".format(cnt))    

    #To evaluate model, select 30% random data from train set
    df = np.hstack([np.array(train_review_list).reshape(-1,1), np.array(train_stars_list).reshape(-1,1)])
    np.random.shuffle(df)
    df = df[:round(len(train_review_list)*0.3)]
    eval_review_list = df[:,0].tolist()
    eval_stars_list = df[:,1].tolist()    

    if method == 'CTF':
        #Token from the original text
        train_all_token_list = [token for text in train_review_list for token in text]
        train_all_token_dict = Counter(train_all_token_list)

        feature_dict = train_all_token_dict.most_common(feature_num)
        feature_list = [f[0] for f in feature_dict]

        train_text_token_list = []
        for text in train_review_list:
            train_text_token_list.append(Counter(text))
        
        eval_text_token_list = []
        for text in eval_review_list:
            eval_text_token_list.append(Counter(text))

        dev_text_token_list = []
        for text in dev_review_list:
            dev_text_token_list.append(Counter(text))

        test_text_token_list = []
        for text in test_review_list:
            test_text_token_list.append(Counter(text))

    elif method == 'DF':
        #Tokens from the set of text
        train_all_token_list = [token for text in train_review_list for token in set(text)]
        train_all_token_dict = Counter(train_all_token_list)

        feature_dict = train_all_token_dict.most_common(feature_num)
        feature_list = [f[0] for f in feature_dict]

        train_text_token_list = []
        for text in train_review_list:
            train_text_token_list.append(Counter(set(text)))
        
        eval_text_token_list = []
        for text in eval_review_list:
            eval_text_token_list.append(Counter(set(text)))

        dev_text_token_list = []
        for text in dev_review_list:
            dev_text_token_list.append(Counter(set(text)))

        test_text_token_list = []
        for text in test_review_list:
            test_text_token_list.append(Counter(set(text)))      

    elif method == 'BAL':
        feature_list = []
        for c in range(5):
            c = c+1
            train_review_list_c = np.asarray(train_review_list)[np.asarray(train_stars_list)==c]
            train_all_token_list = [token for text in train_review_list_c for token in set(text)]
            train_all_token_dict = Counter(train_all_token_list)
            feature_dict = train_all_token_dict.most_common(round(feature_num))
            feature_list_c = [f[0] for f in feature_dict]
            feature_list.append(feature_list_c)
        feature_list = [item for sub in feature_list for item in sub]
        feature_list = np.unique(np.asarray(feature_list))
        feature_num = feature_list.shape[0]

        train_text_token_list = []
        for text in train_review_list:
            train_text_token_list.append(Counter(set(text)))

        eval_text_token_list = []
        for text in eval_review_list:
            eval_text_token_list.append(Counter(set(text)))

        dev_text_token_list = []
        for text in dev_review_list:
            dev_text_token_list.append(Counter(set(text)))

        test_text_token_list = []
        for text in test_review_list:
            test_text_token_list.append(Counter(set(text)))
            
        
    print('Save Data')
    #Save csr_matrix data into libsvm
    train_review_matrix = review_to_csr_matrix(train_text_token_list, feature_list, feature_num)
    train_stars_matrix = stars_to_csr_matrix(train_stars_list, class_num)
    dump_svmlight_file(train_review_matrix, train_stars_matrix, str(method+'_train.libsvm'), multilabel=True)
    print('Data saved: {}'.format(str(method+'_train.libsvm')))

    eval_review_matrix = review_to_csr_matrix(eval_text_token_list, feature_list, feature_num)
    eval_stars_matrix = stars_to_csr_matrix(eval_stars_list, class_num)    
    dump_svmlight_file(eval_review_matrix, eval_stars_matrix, str(method+'_eval.libsvm'), multilabel=True)
    print('Data saved: {}'.format(str(method+'_eval.libsvm')))

    dev_review_matrix = review_to_csr_matrix(dev_text_token_list, feature_list, feature_num)
    dev_stars_matrix = stars_to_csr_matrix(dev_stars_list, class_num)
    dump_svmlight_file(dev_review_matrix, dev_stars_matrix, str(method+'_dev.libsvm'), multilabel=True)
    print('Data saved: {}'.format(str(method+'_dev.libsvm')))

    test_review_matrix = review_to_csr_matrix(test_text_token_list, feature_list, feature_num)
    test_stars_matrix = stars_to_csr_matrix(test_stars_list, class_num)
    dump_svmlight_file(test_review_matrix, test_stars_matrix, str(method+'_test.libsvm'), multilabel=True)
    print('Data saved: {}'.format(str(method+'_test.libsvm')))
