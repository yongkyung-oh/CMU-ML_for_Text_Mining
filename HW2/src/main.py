#!/usr/bin/bash

__author__ = 'YongKyung Oh'

import os
import time
import math

import argparse
import numpy as np
import scipy as sp
import torch

from collections import defaultdict
from scipy.sparse import *
from scipy.sparse import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import utils
import functions

# Define file_path
current_dir = os.getcwd()
train_path = os.path.join(current_dir, 'data', 'train.csv')
dev_path = os.path.join(current_dir, 'data', 'dev.csv')
dev_queries_path = os.path.join(current_dir, 'data', 'dev.queries')
test_path = os.path.join(current_dir, 'data', 'test.csv')
test_queries_path = os.path.join(current_dir, 'data', 'test.queries')


# argparse for parameters
def parse_args():
    parser = argparse.ArgumentParser(description='Python module for Collaborative Filtering')
    parser.add_argument('--Method', type=str, required=True, help='CF methods: user | item | pcc_user | pcc_item | pmf_gd | pmf_torch')
    parser.add_argument('--File', type=str, required=True, help='Select File: dev | test')
    parser.add_argument('--Similarity', type=str, required=False, default='dot', help='Similarity measure: dot | cosine')
    parser.add_argument('--Mean', type=str, required=False, default='mean', help='Mean calculation: mean | weight')
    parser.add_argument('--K', type=int, required=False, default=5, help='KNN parameter (default: 5)')
    parser.add_argument('--Latent', type=int, required=False, default=10, help='PMF latent parameter (default: 10)')
    parser.add_argument('--Lambda', type=float, required=False, default=0.2, help='PMF lambda parameter (default: 0.2)')
    parser.add_argument('--Save', type=str, required=False, default='False', help='Outcome name setup True / False')

    return parser.parse_args()


if __name__ == '__main__':
    t_start = time.perf_counter()
    np.random.seed(0)
    params = parse_args()

    # parameter initialization -----------------------------------------------------------------------------------------
    method_set = ['user', 'item', 'pcc_user', 'pcc_item', 'pmf_gd', 'pmf_torch']
    if params.Method not in method_set:
        raise ValueError('Not valid method')
    method = params.Method

    file_set = ['dev', 'test']
    if params.File not in file_set:
        raise ValueError('Not valid file')
    target_file = params.File

    similarity_set = ['dot', 'cosine']
    if params.Similarity not in similarity_set:
        raise ValueError('Not valid similarity')
    similarity_param = params.Similarity

    mean_set = ['mean', 'weight']
    if params.Mean not in mean_set:
        raise ValueError('Not valid mean method')
    mean_param = params.Mean

    if params.K < 0.0:
        raise ValueError('Not valid parameter')
    K = params.K

    if params.Latent < 0.0:
        raise ValueError('Not valid parameter')
    D = params.Latent

    if params.Lambda < 0.0 or params.Lambda > 1.0:
        raise ValueError('Not valid parameter')
    lmd = params.Lambda

    if params.Save != 'True' and params.Save != 'False':
        raise ValueError('Not valid parameter')
    Save = params.Save

    # data load --------------------------------------------------------------------------------------------------------
    dev_query, dev_rows_cols = utils.load_query_data(dev_queries_path)
    test_query, test_rows_cols = utils.load_query_data(test_queries_path)

    dev_mv_list, dev_us_list = functions.load_csv(dev_path)
    test_mv_list, test_us_list = functions.load_csv(test_path)

    train_data = utils.load_review_data_matrix(train_path)
    train_matrix = train_data.X

    if target_file == 'dev':
        target_mv_list = dev_mv_list
        target_us_list = dev_us_list
        dev_matrix = csr_matrix((dev_query, (dev_rows_cols)), shape=(train_matrix.shape))
        target_matrix = dev_matrix
    elif target_file == 'test':
        target_mv_list = test_mv_list
        target_us_list = test_us_list
        test_matrix = csr_matrix((test_query, (test_rows_cols)), shape=(train_matrix.shape))
        target_matrix = test_matrix

    # experiments ------------------------------------------------------------------------------------------------------
    ## Experiment 1. User-User CF
    if method == 'user':
        q_us_mv_dict = defaultdict(list)
        for k, v in zip(target_us_list, target_mv_list):
            q_us_mv_dict[k].append(v)
        q_us_set = set(target_us_list)
        q_us_set_list = list(q_us_set)

        if similarity_param == 'dot':
            # us_us_dot_sim = train_data.X.dot(train_data.X.T)
            us_us_dot_sim = np.dot(target_matrix, train_matrix.T)
            sim_score = us_us_dot_sim
        elif similarity_param == 'cosine':
            # us_us_cos_sim = pairwise.cosine_similarity(train_data.X, dense_output=False)
            target_matrix_normalized = normalize(target_matrix, axis=1)
            train_matrix_normalized = normalize(train_matrix, axis=1)  # Normailize by row (axis = 1)
            us_us_cos_sim = np.dot(target_matrix_normalized, train_matrix_normalized.T)
            sim_score = us_us_cos_sim

        pcc = False
        predict = functions.user_cf_pred(train_matrix, q_us_set_list, q_us_mv_dict, mean_param, sim_score, K, pcc)

    ## Experiment 2. Item-Item CF
    elif method == 'item':
        q_mv_us_dict = defaultdict(list)
        for k, v in zip(target_mv_list, target_us_list):
            q_mv_us_dict[k].append(v)
        q_mv_set = set(target_mv_list)
        q_mv_set_list = list(q_mv_set)

        if similarity_param == 'dot':
            # mv_mv_dot_sim = train_data.X.T.dot(train_data.X)
            mv_mv_dot_sim = np.dot(target_matrix.T, train_matrix)
            sim_score = mv_mv_dot_sim
        elif similarity_param == 'cosine':
            # mv_mv_cos_sim = pairwise.cosine_similarity(train_data.X.T, dense_output=False)
            target_matrix_normalized = normalize(target_matrix, axis=0)  # Normailize by column (axis = 0)
            train_matrix_normalized = normalize(train_matrix, axis=0)  # Normailize by column (axis = 0)
            mv_mv_cos_sim = np.dot(target_matrix_normalized.T, train_matrix_normalized)
            sim_score = mv_mv_cos_sim

        pcc = False
        predict = functions.item_cf_pred(train_matrix, q_mv_set_list, q_mv_us_dict, mean_param, sim_score, K, pcc)


    ## Experiment 3-1. User-User CF
    elif method == 'pcc_user':
        q_us_mv_dict = defaultdict(list)
        for k, v in zip(target_us_list, target_mv_list):
            q_us_mv_dict[k].append(v)
        q_us_set = set(target_us_list)
        q_us_set_list = list(q_us_set)

        if similarity_param == 'dot':
            # us_us_dot_sim = train_data.X.dot(train_data.X.T)
            target_matrix_std = target_matrix - np.sum(target_matrix, axis=1) / target_matrix.shape[0]
            train_matrix_std = train_matrix - np.sum(train_matrix, axis=1) / train_matrix.shape[0]
            us_us_pcc_sim = np.asarray(np.dot(target_matrix_std, train_matrix_std.T))
            sim_score = us_us_pcc_sim
        elif similarity_param == 'cosine':
            # us_us_cos_sim = pairwise.cosine_similarity(train_data.X, dense_output=False)
            target_matrix_std = target_matrix - np.sum(target_matrix, axis=1) / target_matrix.shape[0]
            target_matrix_normalized = normalize(target_matrix_std, axis=1)  # Normailize by row (axis = 1)
            train_matrix_std = train_matrix - np.sum(train_matrix, axis=1) / train_matrix.shape[0]
            train_matrix_normalized = normalize(train_matrix_std, axis=1)  # Normailize by row (axis = 1)
            us_us_pcc_sim = np.dot(target_matrix_normalized, train_matrix_normalized.T)
            sim_score = us_us_pcc_sim

        pcc = True
        predict = functions.user_cf_pred(train_matrix, q_us_set_list, q_us_mv_dict, mean_param, sim_score, K, pcc)

    ## Experiment 3-2. Item-Item CF
    elif method == 'pcc_item':
        q_mv_us_dict = defaultdict(list)
        for k, v in zip(target_mv_list, target_us_list):
            q_mv_us_dict[k].append(v)
        q_mv_set = set(target_mv_list)
        q_mv_set_list = list(q_mv_set)

        if similarity_param == 'dot':
            # mv_mv_dot_sim = train_data.X.T.dot(train_data.X)
            target_matrix_std = target_matrix - np.sum(target_matrix, axis=0) / target_matrix.shape[1]
            train_matrix_std = train_matrix - np.sum(train_matrix, axis=0) / train_matrix.shape[1]
            mv_mv_pcc_sim = np.asarray(np.dot(target_matrix_std.T, train_matrix_std))
            sim_score = mv_mv_pcc_sim
        elif similarity_param == 'cosine':
            # mv_mv_cos_sim = pairwise.cosine_similarity(train_data.X.T, dense_output=False)
            target_matrix_std = target_matrix - np.sum(target_matrix, axis=0) / target_matrix.shape[1]
            target_matrix_normalized = normalize(target_matrix_std, axis=0)  # Normailize by column (axis = 0)
            train_matrix_std = train_matrix - np.sum(train_matrix, axis=0) / train_matrix.shape[1]
            train_matrix_normalized = normalize(train_matrix_std, axis=0)  # Normailize by column (axis = 0)
            mv_mv_pcc_sim = np.dot(target_matrix_normalized.T, train_matrix_normalized)
            sim_score = mv_mv_pcc_sim

        pcc = True
        predict = functions.item_cf_pred(train_matrix, q_mv_set_list, q_mv_us_dict, mean_param, sim_score, K, pcc)

    ## Experiment 4-1. PMF_gd
    elif method == 'pmf_gd':
        train_data = utils.load_review_data_matrix(train_path, normalize=0)
        U, V, n_iter = functions.matrix_factorization(train_data, D, lmd)

        score_list = []
        predict = defaultdict(dict)
        for mv_id, us_id in zip(target_mv_list, target_us_list):
            #    score = (U[us_id] * V[mv_id].T)[0,0]
            score = (U[us_id]).dot(V[mv_id])
            score_list.append(score)
            predict[mv_id][us_id] = score

        score_min, score_max = min(score_list), max(score_list)

        for mv_id, us_id in zip(target_mv_list, target_us_list):
            score_tmp = predict[mv_id][us_id]
            score_normal = (score_tmp - score_min) / (score_max - score_min) * 4 + 1
#            predict[mv_id][us_id] = int(round(score_normal))
            predict[mv_id][us_id] = float(score_normal)

    ## Experiment 4-2. PMF_torch
    elif method == 'pmf_torch':
        train_data = utils.load_review_data_matrix(train_path, normalize=0)
        if torch.cuda.is_available() == True:
            U, V, n_iter = functions.torch_matrix_factorization_cuda(train_data, D, lmd)
        else:
            U, V, n_iter = functions.torch_matrix_factorization(train_data, D, lmd)

        score_list = []
        predict = defaultdict(dict)
        for mv_id, us_id in zip(target_mv_list, target_us_list):
            #    score = (U[us_id] * V[mv_id].T)[0,0]
            #    score = (U[us_id]).dot(V[mv_id])
            score = torch.sigmoid((U[us_id]).dot(V[mv_id])).squeeze()
            # score = (score*4 + 1).round().tolist()
            score_list.append(score)
            predict[mv_id][us_id] = score.tolist()
        score_min, score_max = min(score_list), max(score_list)

        for mv_id, us_id in zip(target_mv_list, target_us_list):
            score_tmp = predict[mv_id][us_id]
            score_normal = (score_tmp - score_min) / (score_max - score_min) * 4 + 1
#            predict[mv_id][us_id] = int(score_normal.round())
            predict[mv_id][us_id] = float(score_normal)

    # save output ------------------------------------------------------------------------------------------------------
    if Save == 'True':
        output_name = str(target_file + '-predictions.txt')
    else:
        if method in ['user', 'item', 'pcc_user', 'pcc_item']:
            output_name = str(method + '_' + target_file + '_' + similarity_param + '_' + mean_param + '_' + str(K) + '.txt')
        else:
            output_name = str(method + '_' + target_file + '_' + str(D) + '_' + str(lmd) + '_' + str(n_iter+1) + '.txt')



    f = open(output_name, 'w')
    # predict_out = []
    for mv_id, us_id in zip(target_mv_list, target_us_list):
        # predict_out.append(predict[target_us_list[i]][target_mv_list[i]])
        f.write(str(str(predict[mv_id][us_id]) + '\n'))
    f.close()

    # print time  ------------------------------------------------------------------------------------------------------
    t_end = time.perf_counter()
    print('Output Name: {}\n'.format(output_name))
    print('Program Running Time: {}(s)\n'.format(t_end - t_start))

    log_name = str('log_'+output_name)
    f = open(log_name, 'w')
    f.write('Output Name: {}\n'.format(output_name))
    f.write('Program Running Time: {}(s)\n'.format(t_end - t_start))
