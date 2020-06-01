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

def load_csv(path):
    mov_ids, u_ids = [], []
    with open(path) as f:
        for line in f:
            mov_id, u_id = line.split(',')
            mov_id = int(mov_id)
            u_id = int(u_id)
            mov_ids.append(mov_id)
            u_ids.append(u_id)
    return np.array(mov_ids), np.array(u_ids)


def user_cf_pred(train_matrix, q_us_set_list, q_us_mv_dict, mean_param, sim_score, k, pcc):
    predict = defaultdict(dict)
    for q_us_id in q_us_set_list:
        if pcc == False:
            ind = sim_score[q_us_id, :].toarray().argsort()[0][-k-1:][::-1]
        elif pcc == True:
            ind = np.asarray(sim_score[q_us_id, :]).argsort()[-k-1:][::-1]
        knn_ind = ind[1:]
        knn_data = train_matrix[knn_ind,:]
        sim_list = [sim_score[q_us_id, knn_ind] for knn_ind in knn_ind]
        weight_list = sim_list
        if sum(sim_list) != 0:
            weight_list = sim_list / sum(sim_list)
        q_mv_ind = q_us_mv_dict[q_us_id]
        for mv_ind in q_mv_ind:
            if mean_param == 'mean':
    #            predict.append(np.mean(knn_data[:,mv_ind].toarray()))
                score = np.mean(knn_data[:,mv_ind].toarray())
#                predict[mv_ind][q_us_id] = int(round(score)+3)
                predict[mv_ind][q_us_id] = float(score + 3)
            elif mean_param == 'weight':
    #            predict.append(np.dot(weight_list, knn_data[:,mv_ind].toarray())[0])
                score = np.dot(weight_list, knn_data[:,mv_ind].toarray())[0]
#                predict[mv_ind][q_us_id] = int(round(score)+3)
                predict[mv_ind][q_us_id] = float(score + 3)
    return predict


def item_cf_pred(train_matrix, q_mv_set_list, q_mv_us_dict, mean_param, sim_score, k, pcc):
    predict = defaultdict(dict)
    for q_mv_id in q_mv_set_list:
        if pcc == False:
            ind = sim_score[q_mv_id, :].toarray().argsort()[0][-k-1:][::-1]
        elif pcc == True:
            ind = np.asarray(sim_score[q_mv_id, :]).argsort()[-k-1:][::-1]
        knn_ind = ind[1:]
        knn_data = train_matrix[:, knn_ind]
        sim_list = [sim_score[q_mv_id, knn_ind] for knn_ind in knn_ind]
        weight_list = sim_list
        if sum(sim_list) != 0:
            weight_list = sim_list / sum(sim_list)
        q_us_ind = q_mv_us_dict[q_mv_id]
        for us_ind in q_us_ind:
            if mean_param == 'mean':
    #            predict.append(np.mean(knn_data[:,mv_ind].toarray()))
                score = np.mean(knn_data[us_ind, :].toarray().T)
#                predict[q_mv_id][us_ind] = int(round(score) + 3)
                predict[q_mv_id][us_ind] = float(score + 3)
            elif mean_param == 'weight':
    #            predict.append(np.dot(weight_list, knn_data[:,mv_ind].toarray())[0])
                score = np.dot(weight_list, knn_data[us_ind, :].toarray().T)[0]
#                predict[q_mv_id][us_ind] = int(round(score) + 3)
                predict[q_mv_id][us_ind] = float(score + 3)
    return predict


def matrix_factorization(train_data, latent, lmd):
    train_matrix = train_data.X

    us_num = train_matrix.shape[0]
    mv_num = train_matrix.shape[1]

    I = np.zeros(train_matrix.shape)
    row, col = train_data.X.nonzero()
    I[row, col] = 1

#    U, s, V = linalg.svds(train_matrix, k=latent)
#    V = V.T

    U = np.random.rand(us_num, latent)
    V = np.random.rand(mv_num, latent)

    # parameter setup
    iter = 1000
    step = 1e-4
    threshold = 1e-5
    err = np.sum((I * np.asarray(train_matrix - np.dot(U, V.T))) ** 2)

    for i in range(iter):
        A = - (I * np.asarray(train_matrix - np.dot(U, V.T)))
        weight_U = step * (A.dot(V) - lmd * U)
        weight_V = step * (A.T.dot(U) - lmd * V)
        U = U - weight_U
        V = V - weight_V
        new_err = np.sum((I * np.asarray(train_matrix - np.dot(U, V.T))) ** 2)
        err_ratio = abs(float(new_err) - float(err)) / float(err)
        if err_ratio < threshold:
            break
        err = new_err
        if i % 10 == 0: print('Iter: {} | Error: {:.5f} | Error ratio: {:.5f}'.format(i, new_err, err_ratio))

    return U, V, i


def torch_matrix_factorization(train_data, latent, lmd):
    train_matrix = train_data.X
#    train_matrix_normal = normalize(train_matrix, axis=1)  # Normailize by row (axis = 1)
#    train_matrix_normal = normalize(train_matrix_normal, axis=0)  # Normailize by column (axis = 0)

    # Train Matrix to tensor
    train_data.X.data = (train_data.X.data - 1) / 4

    I = np.zeros(train_matrix.shape)
    row, col = train_data.X.nonzero()
    I[row, col] = 1
    train_data_filter = train_data.X.todense() + I - 1
    train_tensor = torch.tensor(train_data_filter)

    # Number of users nad movies, define U and V
    us_num = train_matrix.shape[0]
    mv_num = train_matrix.shape[1]

#    U, s, V = linalg.svds(train_matrix_normal, k=latent)
#    V = V.T

#    us_feature = torch.from_numpy(U).float()
#    us_feature.requires_grad = True
#    mv_feature = torch.from_numpy(V).float()
#    mv_feature.requires_grad = True

    us_feature = torch.randn(us_num, latent, requires_grad=True)
    us_feature.data.mul_(0.01)
    mv_feature = torch.randn(mv_num, latent, requires_grad=True)
    mv_feature.data.mul_(0.01)

    # Train Matrix to tensor
#    values = train_matrix_normal.data
#    indices = np.vstack((train_data.rows, train_data.cols))
#    ind = torch.LongTensor(indices)
#    val = torch.FloatTensor(values)
#    shape = train_matrix_normal.shape
#    train_tensor = torch.sparse.FloatTensor(ind, val, torch.Size(shape))

    # Masked matrix for nonzero
    I = np.zeros(train_matrix.shape)
    row, col = train_data.X.nonzero()
    I[row, col] = 1
    I = torch.FloatTensor(I)

    # Define Loss
    class PMFLoss(torch.nn.Module):
        def __init__(self, lmd_u=0.1, lmd_v=0.1):
            super().__init__()
            self.lmd_u = lmd_u
            self.lmd_v = lmd_v

        def forward(self, matrix, I, u_features, v_features):
            predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))  # Sigmoid of Dot product

            diff = (matrix - predicted) ** 2  # Loss as RMSE
            prediction_error = torch.sum(diff * I)

            u_regularization = self.lmd_u * torch.sum(u_features.norm(dim=1))
            v_regularization = self.lmd_v * torch.sum(v_features.norm(dim=1))

            return prediction_error + u_regularization + v_regularization

    criterion = PMFLoss(lmd_u=lmd, lmd_v=lmd)
    loss = criterion(train_tensor, I, us_feature, mv_feature)
    optimizer = torch.optim.Adam([us_feature, mv_feature], lr=0.01, weight_decay=1e-4)
    threshold = 1e-7

    for step, epoch in enumerate(range(1000)):
        optimizer.zero_grad()
        new_loss = criterion(train_tensor, I, us_feature, mv_feature)
        new_loss.backward()
        optimizer.step()

        loss_ratio = abs(float(new_loss) - float(loss)) / float(loss)
        loss = new_loss
        if step % 10 == 0:
            print('Iter: {} | Loss: {:.5f} | Loss ratio: {:.5f}'.format(step, new_loss, loss_ratio))
            if loss_ratio == 0:
                continue
            elif loss_ratio < threshold:
                break

    U = us_feature
    V = mv_feature

    return U, V, step


def torch_matrix_factorization_cuda(train_data, latent, lmd):
    print('using cuda')
    torch.cuda.init()
    train_matrix = train_data.X

    # Train Matrix to tensor
    train_data.X.data = (train_data.X.data - 1) / 4
    train_tensor = torch.tensor(train_data.X.todense())

    # Number of users nad movies, define U and V
    us_num = train_matrix.shape[0]
    mv_num = train_matrix.shape[1]

    us_feature = torch.randn(us_num, latent, requires_grad=True)
    us_feature.data.mul_(0.01)
    mv_feature = torch.randn(mv_num, latent, requires_grad=True)
    mv_feature.data.mul_(0.01)

    # Train Matrix to tensor

    # Masked matrix for nonzero
    I = np.zeros(train_matrix.shape)
    row, col = train_data.X.nonzero()
    I[row, col] = 1
    I = torch.FloatTensor(I)

    # Define Loss
    class PMFLoss(torch.nn.Module):
        def __init__(self, lmd_u=0.1, lmd_v=0.1):
            super().__init__()
            self.lmd_u = lmd_u
            self.lmd_v = lmd_v

        def forward(self, matrix, I, u_features, v_features):
            predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))  # Sigmoid of Dot product

            diff = (matrix - predicted) ** 2  # Loss as RMSE
            prediction_error = torch.sum(diff * I)

            u_regularization = self.lmd_u * torch.sum(u_features.norm(dim=1))
            v_regularization = self.lmd_v * torch.sum(v_features.norm(dim=1))

            return prediction_error + u_regularization + v_regularization

    cuda = torch.device('cuda')
    train_tensor_cuda = train_tensor.cuda()
    I_cuda = I.cuda()

    us_feature_cuda = torch.randn(us_num, latent, requires_grad=True, device = cuda)
    us_feature_cuda.data.mul_(0.01)
    mv_feature_cuda = torch.randn(mv_num, latent, requires_grad=True, device = cuda)
    mv_feature_cuda.data.mul_(0.01)

    criterion = PMFLoss(lmd_u=lmd, lmd_v=lmd).cuda()
    loss_cuda = criterion(train_tensor_cuda, I_cuda, us_feature_cuda, mv_feature_cuda).cuda()
    optimizer = torch.optim.Adam([us_feature_cuda, mv_feature_cuda], lr=0.01, weight_decay=1e-4)
    threshold = 1e-8

    for step, epoch in enumerate(range(5000)):
        optimizer.zero_grad()
        new_loss = criterion(train_tensor_cuda, I_cuda, us_feature_cuda, mv_feature_cuda).cuda()
        new_loss.backward()
        optimizer.step()

        loss_ratio = abs(float(new_loss) - float(loss_cuda)) / float(loss_cuda)
        loss_cuda = new_loss
        if step%50 == 0:
            print('Iter: {} | Loss: {:.5f} | Loss ratio: {:.5f}'.format(step, new_loss, loss_ratio))
            if loss_ratio == 0:
                continue
            elif loss_ratio < threshold:
                break

    U = us_feature_cuda
    V = mv_feature_cuda

    torch.cuda.empty_cache()

    return U, V, step