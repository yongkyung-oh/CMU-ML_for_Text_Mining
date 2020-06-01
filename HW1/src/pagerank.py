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
threshold = 1e-10

# Define file_path
current_dir = os.getcwd()
transition_path = os.path.join(current_dir,  'data', 'transition.txt')
doc_topic_path = os.path.join(current_dir, 'data', 'doc_topics.txt')
query_topic_path = os.path.join(current_dir, 'data', 'query-topic-distro.txt')
user_topic_path = os.path.join(current_dir, 'data', 'user-topic-distro.txt')
score_data_path = os.path.join(current_dir, 'data', 'indri-lists')

# argparse for parameters
def parse_args():
    parser = argparse.ArgumentParser(description='Python module for Pagerank Score')
    parser.add_argument('--Method', '-M' , type=str, required=True, help='Pagerank methods: GPR/QTSPR/PTSPR')
    parser.add_argument('--Save', type=bool, required=False, default=False, help='Determine json output (default: False)')
    parser.add_argument('--Output', '-O' , type=str, required=False, help='Output file name (default: [Method]_output.json')
    parser.add_argument('--alpha', type=float, required=False, default=0.2, help='Damping parameter (default: 1-alpha = 0.8')
    parser.add_argument('--beta', type=float, required=False, default=0.1, help='TSPR parameter (default: beta = 0.1')

    return parser.parse_args()


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# pagerank algorithm setup
def pagerank_setup(transition_path, doc_topic_path, size):
    M, M_zero = pagerank_function.get_transiton_matrix(transition_path, size)

    # Setup p0
    p0 = np.zeros(size)
    p0 = p0.T
    p0.fill(float(1 / size))

    # Setup pt
    pt_dict = pagerank_function.get_pt(doc_topic_path, size)

    return M, M_zero, p0, pt_dict


if __name__=='__main__':
    t_start = time.perf_counter()
    np.random.seed(0)
    params = parse_args()

    # parameter initialization -----------------------------------------------------------------------------------------
    method_set = ['GPR', 'QTSPR', 'PTSPR']
    if params.Method not in method_set:
        raise ValueError('Not valid method')
    method = params.Method

    save = bool(params.Save)
    if params.Output == None:
        output_name = str(method+'_output.json')
    else:
        output_name = params.Output

    if params.alpha < 0.0:
        raise ValueError('Not valid parameter')
    elif params.beta < 0.0:
        raise ValueError('Not valid parameter')
    elif params.alpha + params.beta > 1.0:
        raise ValueError('Not valid parameter')
    alpha = params.alpha
    beta = params.beta

#    print('------------------------------------------<Get Pagerank Scores>------------------------------------------')
    # pagerank initial setup -------------------------------------------------------------------------------------------
    M, M_zero, p0, pt_dict = pagerank_setup(transition_path, doc_topic_path, size)

    # calculate score using pagerank method ----------------------------------------------------------------------------
    if method == 'GPR':
        GPR = pagerank_function.cal_gpr_score(alpha, M, M_zero, p0, size, threshold)  # Damping factor is 0.8 (1-0.2)
        if save:
            GPR_dict = dict(zip(range(size), GPR))
            GPR_json = json.dumps(GPR_dict)
            f = open(output_name, 'w')
            f.write(GPR_json)
            f.close()

    elif method == 'QTSPR' or method == 'PTSPR':
        TSPR = pagerank_function.cal_tspr_matrix(1 - alpha, beta, M, M_zero, pt_dict, p0, size, topic_size,
                                                 threshold)  # Damping factor is 0.8 (1-0.2)
        if method == 'QTSPR':
            file_path = query_topic_path
            QTSPR_dict = pagerank_function.cal_tspr_score(method, file_path, TSPR, topic_size, query_size)

            if save:
                QTSPT_json = json.dumps(QTSPR_dict, cls=NumpyEncoder)
                f = open(output_name, 'w')
                f.write(QTSPT_json)
                f.close()

        elif method == 'PTSPR':
            file_path = user_topic_path
            PTSPR_dict = pagerank_function.cal_tspr_score(method, file_path, TSPR, topic_size, query_size)

            if save:
                PTSPT_json = json.dumps(PTSPR_dict, cls=NumpyEncoder)
                f = open(output_name, 'w')
                f.write(PTSPT_json)

    t_end = time.perf_counter()

#    print('\n-------------------------------------------<Terminate Program>-------------------------------------------')
    if save: print('Output filename: {}'.format(output_name))
    else: print('{} score calculate'.format(method))
    print('Program Running Time: {}(s)\n'.format(t_end-t_start))
