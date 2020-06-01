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

import pagerank_function


# Initial Parameters
size = 81433
topic_size = 12
query_size = 6

alpha = 0.2
beta = 0.1
threshold = 1e-10

# Define file_path
current_dir = os.getcwd()
transition_path = os.path.join(current_dir,  'data', 'transition.txt')
doc_topic_path = os.path.join(current_dir, 'data', 'doc_topics.txt')
query_topic_path = os.path.join(current_dir, 'data', 'query-topic-distro.txt')
user_topic_path = os.path.join(current_dir, 'data', 'user-topic-distro.txt')
score_data_path = os.path.join(current_dir, 'data', 'indri-lists')

if __name__=='__main__':
    t_start = time.perf_counter()
    np.random.seed(0)

    print('------------------------------------------<Get Pagerank Scores>------------------------------------------')
    # call transition matrix
    M, M_zero = pagerank_function.get_transiton_matrix(transition_path, size)

    # setup p0
    p0 = np.zeros(size)
    p0 = p0.T
    p0.fill(float(1 / size))

    # setup pt
    pt_dict = pagerank_function.get_pt(doc_topic_path, size)

    # gpr score
    print('\n[GPR score]')
    GPR = pagerank_function.cal_gpr_score(alpha, M, M_zero, p0, size, threshold)  # Damping factor is 0.8 (1-0.2)

    output_name = 'GPR.txt'
    output = open(output_name, 'w')

    for i in range(size):
        out = str(i) + ' ' + str(GPR[i])
        output.write(out)
        output.write('\n')

    print('Sample: {} saved.'.format(output_name))

    # tspr score
    print('\n[TSPR score]')
    TSPR = pagerank_function.cal_tspr_matrix(1 - alpha, beta, M, M_zero, pt_dict, p0, size, topic_size,
                                             threshold)  # Damping factor is 0.8 (1-0.2)

    user_id = 2
    query_id = 0  # Python index start from 0 rather than 1

    # QTSPR
    print('\n[QTSPR score]')
    method = 'QTSPR'
    file_path = query_topic_path

    QTSPR_DICT = pagerank_function.cal_tspr_score(method, file_path, TSPR, topic_size, query_size)

    output_name = 'QTSPR-U2Q1.txt'
    output = open(output_name, 'w')

    for i in range(size):
        out = str(i) + ' ' + str(QTSPR_DICT[user_id][:, query_id][i])
        output.write(out)
        output.write('\n')

    print('Sample: {} saved.'.format(output_name))

    # PTSPR
    print('\n[PTSPR score]')
    method = 'PTSPR'
    file_path = user_topic_path

    PTSPR_DICT = pagerank_function.cal_tspr_score(method, file_path, TSPR, topic_size, query_size)

    output_name = 'PTSPR-U2Q1.txt'
    output = open(output_name, 'w')

    for i in range(size):
        out = str(i) + ' ' + str(PTSPR_DICT[user_id][:, query_id][i])
        output.write(out)
        output.write('\n')

    print('Sample: {} saved.'.format(output_name))

    t_end = time.perf_counter()
    print('\n-------------------------------------------<Terminate Program>-------------------------------------------')
    print('\nProgram Running Time: {}(s)\n'.format(t_end-t_start))
