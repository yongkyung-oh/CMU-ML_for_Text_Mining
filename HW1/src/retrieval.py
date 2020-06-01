#!/usr/bin/env

import os
import time
import math
import json

import argparse
import scipy as sp
import numpy as np

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
    parser.add_argument('--Score', '-S', type=str, required=True, help='Retrieval score methods: NS/WS/CM')
    parser.add_argument('--Save', type=bool, required=False, default=True, help='Determine json output (default: False)')
    parser.add_argument('--Output', '-O' , type=str, required=False, help='Output file name (default: [Method]_[Score_method]_output.txt')
    parser.add_argument('--alpha', type=float, required=False, default=0.2, help='Damping parameter (default: 1-alpha = 0.8')
    parser.add_argument('--beta', type=float, required=False, default=0.1, help='TSPR parameter (default: beta = 0.1')
    parser.add_argument('--p_weight', type=float, required=False, default=0.999, help='WS method parameter (default: p_score = 0.999')
    parser.add_argument('--s_weight', type=float, required=False, default=0.001, help='WS method parameter (default: s_score = 0.001')

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

    score_method_set = ['NS', 'WS', 'CM']
    if params.Score not in score_method_set:
        raise ValueError('Not valid score method')
    score_method = params.Score

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

    if params.p_weight < 0.0:
        raise ValueError('Not valid parameter')
    elif params.s_weight < 0.0:
        raise ValueError('Not valid parameter')
    elif params.p_weight + params.s_weight != 1.0:
        raise ValueError('Not valid parameter')
    p_weight = params.p_weight
    s_weight = params.s_weight

    # Initialize all doc list: same as size 81433
    all_doc_id_list = list(range(1, size + 1))

    # Setup output name
    output_name = str(method) + '_' + str(score_method) + '_output.txt'

    # Select CM_method
    if score_method == 'CM': CM_method = 'C07'

    # calculate or import pagerank score using selected method ---------------------------------------------------------
    #print('------------------------------------------<Get Pagerank Scores>------------------------------------------')
    pagerank_start_time = time.perf_counter()
    if method == 'GPR':
        if os.path.exists('GPR_output.json'):
            with open('GPR_output.json', 'r') as f:
                GPR_json_import = json.load(f)
            GPR_import = np.array(list(GPR_json_import.values()))
            GPR_DICT = GPR_import
            print("  {} data imported".format(method))
        else:
            M, M_zero, p0, pt_dict = pagerank_setup(transition_path, doc_topic_path, size)
            GPR = pagerank_function.cal_gpr_score(alpha, M, M_zero, p0, size,
                                                  threshold)  # Damping factor is 0.8 (1-0.2)
            GPR_DICT = GPR
            print("  {} data calculated".format(method))

    elif method == 'PTSPR':
        if os.path.exists('PTSPR_output.json'):
            with open('PTSPR_output.json', 'r') as f:
                PTSPR_json_import = json.load(f)
            PTSPR_import = {int(key): np.asarray(value) for key, value in PTSPR_json_import.items()}
            PTSPR_DICT = PTSPR_import
            print("  {} data imported".format(method))
        else:
            M, M_zero, p0, pt_dict = pagerank_setup(transition_path, doc_topic_path, size)
            TSPR = pagerank_function.cal_tspr_matrix(1 - alpha, beta, M, M_zero, pt_dict, p0, size, topic_size,
                                                     threshold)  # Damping factor is 0.8 (1-0.2)
            file_path = user_topic_path
            PTSPR_dict = pagerank_function.cal_tspr_score(method, file_path, TSPR, topic_size, query_size)
            PTSPR_DICT = PTSPR_dict
            print("  {} data calculated".format(method))
    elif method == 'QTSPR':
        if os.path.exists('QTSPR_output.json'):
            with open('QTSPR_output.json', 'r') as f:
                QTSPR_json_import = json.load(f)
            QTSPR_import = {int(key): np.asarray(value) for key, value in QTSPR_json_import.items()}
            QTSPR_DICT = QTSPR_import
            print("  {} data imported".format(method))
        else:
            M, M_zero, p0, pt_dict = pagerank_setup(transition_path, doc_topic_path, size)
            TSPR = pagerank_function.cal_tspr_matrix(1 - alpha, beta, M, M_zero, pt_dict, p0, size, topic_size,
                                                     threshold)  # Damping factor is 0.8 (1-0.2)
            file_path = query_topic_path
            QTSPR_dict = pagerank_function.cal_tspr_score(method, file_path, TSPR, topic_size, query_size)
            QTSPR_DICT = QTSPR_dict
            print("  {} data calculated".format(method))
    pagerank_time = time.perf_counter()-pagerank_start_time
    #print('Pagerank score get time: {} (s)'.format(time.perf_counter() - pagerank_time))

    # calculate information retrieval score using selected method ------------------------------------------------------
    retrieval_start_time = time.perf_counter()

    output = open(output_name, 'w')
    for file_name in os.listdir(score_data_path):
        file_id = file_name.split(".")[0]

        user_id = int(file_id.split("-")[0])
        query_id = int(file_id.split("-")[1])

        file_path = os.path.join(score_data_path, file_name)

        with open(file_path, 'r') as f:
            IR_doc_id_list = []

            # For all doc list, insert all scores from pagerank
            # If there is no IR score from indri list, substitute None
            pagerank_score_dict_all = {doc_id: None for doc_id in all_doc_id_list}
            search_relevance_score_dict_all = {doc_id: None for doc_id in all_doc_id_list}

            line = f.readline().strip()  # Read line 1 by 1

            # Read the one of the indri list
            # If it has IR score, then save it
            while line != '':
                IR_doc_id = int(line.split()[2])
                IR_doc_id_list.append(IR_doc_id)
                search_relevance_score = float(line.split()[4])

                search_relevance_score_dict_all[IR_doc_id] = search_relevance_score
                line = f.readline().strip()  # Read line 1 by 1

            search_relevance_score_dict = {doc_id: search_relevance_score_dict_all[doc_id] for doc_id in IR_doc_id_list}

            # Save the combined score as dictionary
            combined_score_dict = dict()
            if method == 'GPR':
                pagerank_score_dict = {doc_id: GPR_DICT[doc_id - 1] for doc_id in IR_doc_id_list}
            elif method == 'QTSPR':
                pagerank_score_dict = {doc_id: QTSPR_DICT[user_id][:, query_id - 1][doc_id - 1] for doc_id in
                                       IR_doc_id_list}
            elif method == 'PTSPR':
                pagerank_score_dict = {doc_id: PTSPR_DICT[user_id][:, query_id - 1][doc_id - 1] for doc_id in
                                       IR_doc_id_list}

            pagerank_score = np.asarray(list(pagerank_score_dict.values()))
            search_relevance_score = np.asarray(list(search_relevance_score_dict.values()))

            # Implement the method selection of score combination
            if score_method == 'NS':
                combined_score = pagerank_score
            elif score_method == 'WS':
                combined_score = p_weight * pagerank_score + s_weight * search_relevance_score
            elif score_method == 'CM':
                # Weighted sum method comparisons

                # Customized method
                # Using Normalized value, ignore 0 and -inf (substitute to threshold)
                pagerank_score_normalized = (pagerank_score - min(pagerank_score)) / (
                            max(pagerank_score) - min(pagerank_score))
                pagerank_score_normalized = np.where(pagerank_score_normalized == 0, threshold,
                                                     pagerank_score_normalized)
                pagerank_score_normalized = np.where(pagerank_score_normalized == -math.inf, threshold,
                                                     pagerank_score_normalized)

                valid_search_relevance_score = [s for s in search_relevance_score if
                                                s != -math.inf]  # Ignore invalid value
                search_relevance_score_normalized = (search_relevance_score - min(valid_search_relevance_score)) / (
                            max(valid_search_relevance_score) - min(valid_search_relevance_score))
                search_relevance_score_normalized = np.where(search_relevance_score_normalized == 0, threshold,
                                                             search_relevance_score_normalized)
                search_relevance_score_normalized = np.where(search_relevance_score_normalized == -math.inf, threshold,
                                                             search_relevance_score_normalized)

                if CM_method == 'C00':
                    combined_coeff = np.random.dirichlet(abs(search_relevance_score_normalized)) # C00
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C01':
                    combined_coeff = abs(abs(pagerank_score_normalized)+abs(search_relevance_score_normalized)) # C01
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C02':
                    combined_coeff = abs(abs(pagerank_score_normalized) - abs(search_relevance_score_normalized))  # C02
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C03':
                    combined_coeff = sp.mean([abs(pagerank_score_normalized), abs(search_relevance_score_normalized)],
                                             axis=0)  # C03
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C04':
                    combined_coeff = np.amax([abs(pagerank_score_normalized), abs(search_relevance_score_normalized)],
                                             axis=0)  # C04
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C05':
                    combined_coeff = sp.stats.gmean(
                        [abs(pagerank_score_normalized), abs(search_relevance_score_normalized)], axis=0)  # C05
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C06':
                    combined_coeff = sp.stats.hmean(
                        [abs(pagerank_score_normalized), abs(search_relevance_score_normalized)], axis=0)  # C06
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C07':
                    combined_coeff = - np.exp(abs(pagerank_score_normalized)) + np.exp(
                        abs(search_relevance_score_normalized))  # C07
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C08':
                    combined_coeff = + np.exp(abs(pagerank_score_normalized)) + np.exp(
                        abs(search_relevance_score_normalized))  # C08
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C09':
                    combined_coeff = - np.log(abs(pagerank_score_normalized)) + np.log(
                        abs(search_relevance_score_normalized))  # C09
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)
                elif CM_method == 'C10':
                    combined_coeff = - np.log(abs(pagerank_score_normalized)) - np.log(
                        abs(search_relevance_score_normalized))  # C10
                    combined_coeff = normalize(combined_coeff.reshape(1, -1), norm='l1').reshape(-1)

                combined_score = search_relevance_score_normalized + combined_coeff

            # Using dictionay is too slow
            # combined_score_dict_sorted = {key: value for key, value in sorted(combined_score_dict.items(), key=lambda item: item[1], reverse=True)} # sort the doc_id:combined score dictionary
            # combined_score_dict_sorted_500 = dict(zip(list(combined_score_dict_sorted.keys())[0:500], list(combined_score_dict_sorted.values())[0:500])) # select top 500 data to save

            # combined_score_dict = dict(zip(IR_doc_id_list, combined_score))
            # OUTPUT = np.empty(len(IR_doc_id_list), dtype = [('Id', 'U10'), ('Score', 'f8')])
            # OUTPUT['Id'] = np.asarray(list(combined_score_dict.keys()))
            # OUTPUT['Score'] = np.asarray(list(combined_score_dict.values()))

            # Using structured numpy array instead
            OUTPUT = np.empty(len(IR_doc_id_list), dtype=[('Id', 'U10'), ('Score', 'f8')])
            OUTPUT['Id'] = np.asarray(IR_doc_id_list)
            OUTPUT['Score'] = np.asarray(combined_score)

            OUTPUT_sorted = OUTPUT.copy()
            OUTPUT_sorted[::-1].sort(order='Score')

            # Output file write
            for i in range(len(IR_doc_id_list)):
                out_doc_id = OUTPUT_sorted['Id'][i]
                out_score = OUTPUT_sorted['Score'][i]

                out = str(user_id) + '-' + str(query_id) + ' Q0 ' + str(out_doc_id) + ' ' + str(i + 1) + ' ' + str(
                    out_score) + ' yongkyung'
                output.write(out)
                output.write('\n')

            # print('{}: done'.format(file_id))
    output.close()
    retrieval_time = time.perf_counter() - retrieval_start_time
    #print('Information retrieval get time: {} (s)'.format(retrieval_time))

    t_end = time.perf_counter()

    #print('\n-------------------------------------------<Terminate Program>-------------------------------------------')
    if save: print('{}: {} sec for pagerank, {} sec for retrieval'.format(output_name, pagerank_time, retrieval_time))
    print('Program Running Time: {}(s)\n'.format(t_end-t_start))
