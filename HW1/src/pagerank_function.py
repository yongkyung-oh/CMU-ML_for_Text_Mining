#!/usr/bin/env

import os
import time
import math
import json

import argparse
import numpy as np
import scipy as sp

from scipy.sparse import *
from sklearn.preprocessing import normalize


# get transition matrix
def get_transiton_matrix(transition_path, size):
    # Read transition.txt
    row = []
    col = []
    val = []

    with open(transition_path, 'r') as f:
        line = f.readline().strip()  # Read line 1 by 1
        while line != '':
            row.append(int(line.split()[0]) - 1)  # Python index start from 0 rather than 1
            col.append(int(line.split()[1]) - 1)  # Python index start from 0 rather than 1
            val.append(int(line.split()[2]))

            line = f.readline().strip()

    row = np.array(row, dtype='int')
    col = np.array(col, dtype='int')
    val = np.array(val, dtype='int')

    M = csr_matrix((val, (row, col)), shape=(size, size))

    # Normalization using nonzero row
    row_nz_id = M.nonzero()[0]
    M_row_sum = np.array(np.sum(M, axis=1))[:, 0]
    # M_col_sum = np.array(np.sum(M, axis = 0))[0, :]

    M.data = M.data / M_row_sum[row_nz_id]

    # Substitute zero-sum row to 1/n
    row_zero = np.array(np.where(M_row_sum == 0)[0], dtype=int)
    col_zero = np.zeros(shape=row_zero.shape)
    val_zero = np.zeros(shape=row_zero.shape)
    val_zero.fill(float(1 / size))

    M_zero = csr_matrix((val_zero, (row_zero, col_zero)), shape=(size, 1))
    M_zero = csr_matrix.transpose(M_zero)

    return M, M_zero


# get pt
def get_pt(doc_topic_path, size):
    # Setup pt as dict
    # Create topic-doc dictionary
    pt_dict = dict()

    # doc_topic_dict = dict()
    topic_doc_dict = dict()

    doc_id = []
    topic_id = []

    with open(doc_topic_path, 'r') as f:
        line = f.readline().strip()  # Read line 1 by 1
        while line != '':
            doc = int(line.split()[0])
            doc_id.append(doc)
            topic = int(line.split()[1])
            topic_id.append(topic)

            if topic in topic_doc_dict:
                topic_doc_dict[topic].append(doc)
            else:
                topic_doc_dict[topic] = [doc]

            line = f.readline().strip()

    for topic, doc_id_list in topic_doc_dict.items():
        pt = np.zeros(size)
        pt = pt.transpose()
        # print(topic, size)
        doc_size = len(doc_id_list)
        for doc in doc_id_list:
            pt[doc - 1] = 1 / float(doc_size)

        pt_dict[topic] = pt

    return pt_dict


# gpr score calculation
def cal_gpr_score (alpha, M, M_zero, p0, size, threshold):
    row_zero = M_zero.nonzero()[1]

    gpr = np.zeros(size)
    gpr = gpr.transpose()
    gpr.fill(1)

    gpr_zero = np.zeros(size)
    gpr_zero[row_zero] = 1

    gpr_diff = np.linalg.norm(gpr)

    cnt=0
    while (gpr_diff > threshold):
        gpr_upd = (1-alpha) * csr_matrix.transpose(M) * gpr + (1-alpha) * M_zero * gpr * gpr_zero + alpha * p0
        gpr_diff = np.linalg.norm(gpr-gpr_upd)
        gpr = gpr_upd
        cnt=cnt+1
#    print('  Converged {} iterations'.format(cnt))

    gpr = normalize(gpr.reshape(1, -1), norm='l1').reshape(-1) # Normalize due to truncated error
    return gpr


# tspr score matrix calculation
def cal_tspr_matrix(alpha, beta, M, M_zero, pt_dict, p0, size, topic_size, threshold):
    row_zero = M_zero.nonzero()[1]
    tspr_matrix = np.zeros(shape=(size, topic_size))

    for topic_idx in range(1, 13):
        tspr = np.zeros(size)
        tspr = tspr.transpose()
        tspr.fill(float(1 / size))

        tspr_zero = np.zeros(size)
        tspr_zero[row_zero] = 1

        tspr_diff = np.linalg.norm(tspr)

        cnt = 0
        while (tspr_diff > threshold):
            tspr_upd = alpha * csr_matrix.transpose(M) * tspr + alpha * M_zero * tspr * tspr_zero + beta * pt_dict[
                topic_idx] + (1 - alpha - beta) * p0
            tspr_diff = np.linalg.norm(tspr - tspr_upd)
            tspr = tspr_upd
            cnt = cnt + 1
#        print('  Topic {}: Converged {} iterations'.format(topic_idx, cnt))

        tspr = normalize(tspr.reshape(1, -1), norm='l1').reshape(-1)  # Normalize due to truncated error

        tspr_matrix[:, topic_idx-1] = tspr  # Python index start from 0 rather than 1
    return tspr_matrix


# tspr score calculation for PTSPR / QTSPR
def cal_tspr_score(method, file_path, TSPR, topic_size, query_size):
    user_query_dict = dict()
    user_coeff_dict = dict()

    user_id = []
    query_id = []

    with open(file_path, 'r') as f:
        line = f.readline().strip() #Read line 1 by 1
        while line != '':
            user = int(line.split()[0])
            user_id.append(user)
            query = int(line.split()[1])
            query_id.append(query)

            if user in user_query_dict:
                user_query_dict[user].append(query)
            else:
                user_query_dict[user] = [query]

            coeff = np.zeros(topic_size)
            coeff.transpose()

            for topic_idx in range(1,13):
                coeff[topic_idx-1] = float(line.split()[1+topic_idx].split(':')[1]) # Python index start from 0 rather than 1

            if user in user_coeff_dict:
                user_coeff_dict[user][:, query-1] = coeff
            else:
                user_coeff_dict[user] = np.zeros(shape=(topic_size, query_size))
                user_coeff_dict[user][:, query-1] = coeff # Python index start from 0 rather than 1

            line = f.readline().strip()

    tspr_score_matrix_dict = dict()
    for user_id, coeff in user_coeff_dict.items():
        tspr_score_matrix_dict[user_id] = np.dot(TSPR, coeff)

    return tspr_score_matrix_dict