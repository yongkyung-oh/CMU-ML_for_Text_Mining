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
from sklearn.datasets import load_svmlight_file
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

lmd = 0.01
alpha = 0.0025
beta = alpha/1000
threshold = 1e-8
batch_size = 256

#Define functions
def stars_to_csr_matrix(stars_list, class_num):
    stars_row = range(len(stars_list))
    stars_col = [s-1 for s in stars_list]
    stars_data = [1] * len(stars_list)
    stars_matrix = csr_matrix((stars_data, (stars_row, stars_col)), shape=(len(stars_list), class_num)) 
    
    return stars_matrix

def loss(X_eval, Y_eval, W, lmd):
    w_exp = np.exp(X_eval * W)  
    w_exp_sum = np.sum(w_exp, axis=1).reshape(-1,1)
    p1 = w_exp / w_exp_sum
    p2 = Y_eval.multiply(p1)
    p_sum = np.sum(np.log(np.sum(p2, axis=1)))
    w_sum = np.sum(np.square(W))
    loss = p_sum - (lmd/2)*w_sum
    return loss

def predict(X_eval, W):
    w_exp = np.exp(X_eval * W)  
    w_exp_sum = np.sum(w_exp, axis=1).reshape(-1,1)
    p1 = w_exp / w_exp_sum

    pred_hard = p1.argmax(axis=1)+1
    pred_soft = p1.dot([1,2,3,4,5]).T
    return np.asarray(pred_hard), np.asarray(pred_soft)

def cal_accuracy(y_true, y_pred):
    cnt = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==yp:
            cnt += 1
    acc = cnt / len(y_true)
    return acc

def cal_rmse(y_true, y_pred):
    err = 0
    for yt, yp in zip(y_true, y_pred):
        err += (yt-yp)**2
    rmse = math.sqrt(err/len(y_true))
    return rmse

def save_result(out, pred_hard, pred_soft):
    with open(out, 'w') as f:
        for h, s in zip(pred_hard, pred_soft):
            f.write(str(h.item())+' '+str(s.item())+'\n')


if __name__ == '__main__':
    method = str(sys.argv[1]) #[CTF or DF or BAL]
    method_set = ['CTF', 'DF', 'BAL']
    if method not in method_set:
        raise(ValueError('Not Valid method'))
    print('{} Start'.format(method))

    train_file = str(method+'_train.libsvm')
    eval_file = str(method+'_eval.libsvm')
    dev_file = str(method+'_dev.libsvm')
    test_file = str(method+'_test.libsvm')
    
    if os.path.exists(train_file) and os.path.exists(eval_file) and os.path.exists(dev_file) and os.path.exists(test_file):
        print('All files are prepared')
    else:
        raise(ValueError('Preprocess is required'))

    #Load data
    X_train, y_train = load_svmlight_file(train_file)
    X_eval, y_eval = load_svmlight_file(eval_file)
    X_dev, _ = load_svmlight_file(dev_file)
    X_test, _ = load_svmlight_file(test_file)

    Y_train = stars_to_csr_matrix(y_train, class_num)
    Y_eval = stars_to_csr_matrix(y_eval, class_num)

    n = X_train.shape[0]
    idx = np.array(range(n))
    np.random.shuffle(idx)

    W = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(round(n/batch_size)):
        gamma = alpha / (1 + beta * i)
        batch_idx = idx[batch_size*i:batch_size*(i+1)]
        x_i = X_train[batch_idx]
        y_i = Y_train[batch_idx]

        w_exp = np.exp(x_i * W)  
        w_exp_sum = np.sum(w_exp, axis=1).reshape(-1,1)
        W_g = x_i.T * (y_i - w_exp / w_exp_sum) 
        W_new = W + gamma * W_g

        old_loss = loss(X_eval, Y_eval, W, lmd)
        new_loss = loss(X_eval, Y_eval, W_new, lmd)
        diff = abs((old_loss-new_loss)/old_loss)
        if diff < threshold:
            break
        if i % 100 == 0:
            print('Epoch: {} | Loss: {:.5f} | Diff: {:.7f}'.format(i, old_loss, diff))
        W = W_new    
    
    #Save Weight
    weight_dump = str(method+'_weight.pkl')
    with open(weight_dump, 'wb') as f:
        pickle.dump(W, f)
        f.close()
    weight_dump = str(method+'_weight.pkl')
    with open(weight_dump, 'rb') as f:
        W_loaded = pickle.load(f)
        f.close()        

    pred_train_hard, pred_train_soft = predict(X_train, W_loaded)
    pred_eval_hard, pred_eval_soft = predict(X_eval, W_loaded)
    pred_dev_hard, pred_dev_soft = predict(X_dev, W_loaded)
    pred_test_hard, pred_test_soft = predict(X_test, W_loaded)        

    cal_acc_train = cal_accuracy(y_train, pred_train_hard)
    print('{} LR train accuracy: {}'.format(method, cal_acc_train))
    cal_rmse_train = cal_rmse(y_train, pred_train_soft)
    print('{} LR train rmse: {}'.format(method, cal_rmse_train))        

    cal_acc_eval = cal_accuracy(y_eval, pred_eval_hard)
    print('{} LR eval accuracy: {}'.format(method, cal_acc_eval))
    cal_rmse_eval = cal_rmse(y_eval, pred_eval_soft)
    print('{} LR eval rmse: {}'.format(method, cal_rmse_eval))        

    SVM_classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, class_weight='balanced')
    # Train the classifier
    SVM_classifier.fit(X_train, y_train)
    y_train_pred = SVM_classifier.predict(X_train)
    y_eval_pred = SVM_classifier.predict(X_eval)

    cal_acc_train = cal_accuracy(y_train, y_train_pred)
    print('{} SVM train accuracy: {}'.format(method, cal_acc_train))
    cal_acc_eval = cal_accuracy(y_eval, y_eval_pred)
    print('{} SVM eval accuracy: {}'.format(method, cal_acc_eval))        

    #save_result(str(method+'_train-predictions.txt'), pred_train_hard, pred_train_soft)
    #save_result(str(method+'_eval-predictions.txt'), pred_eval_hard, pred_eval_soft)
    save_result(str(method+'_dev-predictions.txt'), pred_dev_hard, pred_dev_soft)
    save_result(str(method+'_test-predictions.txt'), pred_test_hard, pred_test_soft)
    
    print('Output saved')