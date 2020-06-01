import os
import sys
import time

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix, diags, identity

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import conjugateGradient as cg


# Set Constants
DATA_SET = ['covtype', 'realsim']

LAMBDA_DICT = {'covtype': 3631.3203125,
               'realsim': 7230.8750000
              }
LOSS_VAL_DICT = {'covtype': 2541.664519,
                 'realsim':  669.664812
                }

# Set paths
Src_path = os.getcwd() # Code/src
Par_path = os.path.abspath(os.path.join(Src_path, os.pardir)) # Code
Data_path = os.path.abspath(os.path.join(Par_path, 'data')) # Code/data

if not os.path.exists('out'):
    os.mkdir('out')
    print('make output folder')

# Define functions
def get_gradient(X, y, w, N, batch_size, lambda_val):
    A = np.array(X.dot(w))
    I = np.array(range(len(A)))
    I = I[1-y*A > 0]
    X_I = X[I]
    y_I = y[I]
    grad = w / batch_size + 2 * lambda_val / N * X_I.transpose().dot((X_I.dot(w) - y_I))
    return grad

def get_loss(X, y, w, N):
    A = np.array(X.dot(w))
    Y_val = 1-y*A
    Y_val[Y_val < 0] = 0
    #loss = 1/2 * w.dot(w) + np.sum(np.array(Y_val)**2)/N
    loss = np.sum(np.array(Y_val)**2)/N
    return loss

def get_accuracy(X, y, w, N):
    A = np.array(X.dot(w))
    y_predict = np.ones(len(A)) * -1
    y_predict[np.array(A>0)] = 1
    accuracy = float(np.sum(y_predict==y))/len(y)
    return accuracy


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    idx = np.array(range(X.shape[0]))
    np.random.shuffle(idx)
    n_minibatches = idx.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch_idx = idx[i * batch_size:(i + 1) * batch_size]
        X_mini = X[mini_batch_idx]
        y_mini = y[mini_batch_idx]
        mini_batches.append((X_mini, y_mini))
    if idx.shape[0] % batch_size != 0:
        mini_batch = idx[i * batch_size:idx.shape[0]]
        X_mini = X[mini_batch_idx]
        y_mini = y[mini_batch_idx]
        mini_batches.append((X_mini, y_mini))
    return mini_batches


def main():
    # read the train file from first arugment
    train_file = sys.argv[1]

    # read the test file from second argument
    test_file = sys.argv[2]

    # Check input arguments
    if os.path.basename(train_file) == 'covtype.scale.trn.libsvm' and os.path.basename(test_file) == 'covtype.scale.tst.libsvm':
        DATA = 'covtype'
    elif os.path.basename(train_file) == 'realsim.scale.trn.libsvm' and os.path.basename(test_file) == 'realsim.scale.tst.libsvm':
        DATA = 'realsim'
    else:
        raise(ValueError('Invalid Data'))

    # You can use load_svmlight_file to load data from train_file and test_file
    # X_train, y_train = load_svmlight_file(train_file)

    X_train, y_train = load_svmlight_file(train_file)
    X_test, y_test = load_svmlight_file(test_file)

    # You can use cg.ConjugateGradient(X, I, grad, lambda_)

    lambda_val = LAMBDA_DICT[DATA]
    true_val = LOSS_VAL_DICT[DATA]

    print('========= Date: {} ========='.format(DATA))

    # Pegasos(mini_SGD)
    print('========= Pegasos solver =========')

    # Set parameters
    epoch = 2000
    batch_size = 1000
    lr = 0.001
    beta = lr/100

    N = X_train.shape[0]
    w = np.zeros((X_train.shape[1]))
    A = np.array(X_train.dot(w))
    I = np.array(range(len(A)))
    I = I[1 - y_train * A > 0]

    Pegasos_grad = []
    Pegasos_loss = []
    Pegasos_rel = []
    Pegasos_acc = []
    Pegasos_time = []

    start_time = time.perf_counter()
    for i in range(epoch + 1):
        gamma_t = lr / (1 + beta * i)
        #mini_batches = create_mini_batches(X_train, y_train, batch_size)

        grad_norm = []
        total_loss = []

        # Iterate mini-batch SGD (1 batch for 1 epoch)
        # for batch in mini_batches:
        idx = np.array(range(X_train.shape[0]))
        np.random.shuffle(idx)
        batch_idx = idx[0:batch_size]
        X = X_train[batch_idx]
        y = y_train[batch_idx]

        grad = get_gradient(X, y, w, N, batch_size, lambda_val)
        w = w - gamma_t * grad

        grad_norm.append(np.linalg.norm(grad))
        total_loss.append(get_loss(X, y, w, N))

        # Calculate target values
        func_val = 1 / 2 * w.dot(w) + lambda_val * np.sum(total_loss) * (X_train.shape[0] / batch_size)
        relative_val = (func_val - true_val) / true_val

        grad_total = np.sum(grad_norm) * (X_train.shape[0] / batch_size)
        test_acc = get_accuracy(X_test, y_test, w, N)

        Pegasos_grad.append(grad_total)
        Pegasos_loss.append(np.sum(total_loss))
        Pegasos_rel.append(relative_val)
        Pegasos_acc.append(test_acc)
        Pegasos_time.append(time.perf_counter() - start_time)

        if i % 200 == 0 or i == epoch:
            print('Epoch: {} with Loss: {:.4f}'.format(i, np.sum(total_loss)))
            print('  Grad: {:.4f} | Rel: {:.4f} | Acc: {:.4f} '.format(grad_total, relative_val, test_acc))

    # Save output figures
    plt.figure()
    plt.plot(Pegasos_time, Pegasos_grad)
    plt.xlabel('time')
    plt.ylabel('grad')
    plt.title('{}_Pegasos: grad over time'.format(DATA))
    plt.savefig('out/{}_Pegasos_grad.png'.format(DATA))

    #plt.figure()
    #plt.plot(Pegasos_time, Pegasos_loss)
    #plt.xlabel('time')
    #plt.ylabel('loss')
    #plt.title('{}_Pegasos: loss over time'.format(DATA))
    #plt.savefig('out/{}_Pegasos_loss.png'.format(DATA))

    plt.figure()
    plt.plot(Pegasos_time, Pegasos_rel)
    plt.xlabel('time')
    plt.ylabel('relative value')
    plt.title('{}_Pegasos: relative value over time'.format(DATA))
    plt.savefig('out/{}_Pegasos_relative.png'.format(DATA))

    plt.figure()
    plt.plot(Pegasos_time, Pegasos_acc)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.title('{}_Pegasos: accuracy over time'.format(DATA))
    plt.savefig('out/{}_Pegasos_accuracy.png'.format(DATA))


    # SGD
    print('========== SGD solver ==========')

    # Set parameters
    epoch = 200
    batch_size = 1000
    lr = 0.001
    beta = lr/100

    N = X_train.shape[0]
    w = np.zeros((X_train.shape[1]))
    A = np.array(X_train.dot(w))
    I = np.array(range(len(A)))
    I = I[1-y_train*A > 0]

    SGD_grad = []
    SGD_loss = []
    SGD_rel = []
    SGD_acc = []
    SGD_time = []

    start_time = time.perf_counter()
    for i in range(epoch+1):
        gamma_t = lr / (1 + beta * i)
        mini_batches = create_mini_batches(X_train, y_train, 1000)

        grad_norm = []
        total_loss = []

        # Iterate mini-batch SGD (all batches in 1 epoch)
        for batch in mini_batches:
            X, y = batch
            grad = get_gradient(X, y, w, N, batch_size, lambda_val)
            w = w - gamma_t * grad

            grad_norm.append(np.linalg.norm(grad))
            total_loss.append(get_loss(X, y, w, N))

        # Calculate target values
        func_val = 1 / 2 * w.dot(w) + lambda_val * np.sum(total_loss)
        relative_val = (func_val - true_val) / true_val

        grad_total = np.sum(grad_norm)
        test_acc = get_accuracy(X_test, y_test, w, N)

        SGD_grad.append(grad_total)
        SGD_loss.append(np.sum(total_loss))
        SGD_rel.append(relative_val)
        SGD_acc.append(test_acc)
        SGD_time.append(time.perf_counter() - start_time)

        if i % 20 == 0 or i == epoch:
            print('Epoch: {} with Loss: {:.4f}'.format(i, np.sum(total_loss)))
            print('  Grad: {:.4f} | Rel: {:.4f} | Acc: {:.4f} '.format(grad_total, relative_val, test_acc))

    # Save output figures
    plt.figure()
    plt.plot(SGD_time, SGD_grad)
    plt.xlabel('time')
    plt.ylabel('grad')
    plt.title('{}_SGD: grad over time'.format(DATA))
    plt.savefig('out/{}_SGD_grad.png'.format(DATA))

    #plt.figure()
    #plt.plot(SGD_time, SGD_loss)
    #plt.xlabel('time')
    #plt.ylabel('loss')
    #plt.title('{}_SGD: loss over time'.format(DATA))
    #plt.savefig('out/{}_SGD_loss.png'.format(DATA))

    plt.figure()
    plt.plot(SGD_time, SGD_rel)
    plt.xlabel('time')
    plt.ylabel('relative value')
    plt.title('{}_SGD: relative value over time'.format(DATA))
    plt.savefig('out/{}_SGD_relative.png'.format(DATA))

    plt.figure()
    plt.plot(SGD_time, SGD_acc)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.title('{}_SGD: accuracy over time'.format(DATA))
    plt.savefig('out/{}_SGD_accuracy.png'.format(DATA))


    # Newton
    print('========== Newton solver ==========')

    # Set parameters
    epoch = 50
    batch_size = 1  # Ignore mini-batch
    lr = 0.001
    # beta = lr/100

    N = X_train.shape[0]
    w = np.zeros((X_train.shape[1]))
    A = np.array(X_train.dot(w))
    I = np.array(range(len(A)))
    I = I[1 - y_train * A > 0]

    Newton_grad = []
    Newton_loss = []
    Newton_rel = []
    Newton_acc = []
    Newton_time = []

    start_time = time.perf_counter()
    for i in range(epoch+1):
        # gamma_t = lr / (1+beta*i)

        grad_norm = []
        total_loss = []

        # Conduct Newton & conjugate Gradient
        X, y = X_train, y_train
        grad = get_gradient(X, y, w, N, batch_size, lambda_val)

        d, _ = cg.conjugateGradient(X, I, grad, lambda_val)
        w = w + d

        grad_norm.append(np.linalg.norm(grad))
        total_loss.append(get_loss(X, y, w, N))

        # Calculate target values
        func_val = 1 / 2 * w.dot(w) + lambda_val * np.sum(total_loss)
        relative_val = (func_val - true_val) / true_val

        grad_total = np.mean(grad_norm)
        test_acc = get_accuracy(X_test, y_test, w, N)

        Newton_grad.append(grad_total)
        Newton_loss.append(np.sum(total_loss))
        Newton_rel.append(relative_val)
        Newton_acc.append(test_acc)
        Newton_time.append(time.perf_counter() - start_time)

        if i % 5 == 0 or i == epoch:
            print('Epoch: {} with Loss: {:.4f}'.format(i, np.sum(total_loss)))
            print('  Grad: {:.4f} | Rel: {:.4f} | Acc: {:.4f} '.format(grad_total, relative_val, test_acc))

    plt.figure()
    plt.plot(Newton_time, Newton_grad)
    plt.xlabel('time')
    plt.ylabel('grad')
    plt.title('{}_Newton: grad over time'.format(DATA))
    plt.savefig('out/{}_Newton_grad.png'.format(DATA))

    #plt.figure()
    #plt.plot(Newton_time, Newton_loss)
    #plt.xlabel('time')
    #plt.ylabel('loss')
    #plt.title('{}_Newton: loss over time'.format(DATA))
    #plt.savefig('out/{}_Newton_loss.png'.format(DATA))

    plt.figure()
    plt.plot(Newton_time, Newton_rel)
    plt.xlabel('time')
    plt.ylabel('relative value')
    plt.title('{}_Newton: relative value over time'.format(DATA))
    plt.savefig('out/{}_Newton_relative.png'.format(DATA))

    plt.figure()
    plt.plot(Newton_time, Newton_acc)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.title('{}_Newton: accuracy over time'.format(DATA))
    plt.savefig('out/{}_Newton_accuracy.png'.format(DATA))

    # Save all methods in one figure
    plt.figure()
    plt.plot(np.log(Pegasos_time), Pegasos_grad)
    plt.plot(np.log(SGD_time), SGD_grad)
    plt.plot(np.log(Newton_time), Newton_grad)
    plt.xlabel('log(time)')
    plt.ylabel('grad')
    plt.legend(['Pegasos', 'SGD', 'Newton'])
    plt.title('{}_Result: grad over time'.format(DATA))
    plt.savefig('out/{}_Result_grad.png'.format(DATA))

    #plt.figure()
    #plt.plot(np.log(Pegasos_time), Pegasos_loss)
    #plt.plot(np.log(SGD_time), SGD_loss)
    #plt.plot(np.log(Newton_time), Newton_loss)
    #plt.xlabel('log(time)')
    #plt.ylabel('loss')
    #plt.legend(['Pegasos', 'SGD', 'Newton'])
    #plt.title('{}_Result: loss over time'.format(DATA))
    #plt.savefig('out/{}_Result_loss.png'.format(DATA))

    plt.figure()
    plt.plot(np.log(Pegasos_time), Pegasos_rel)
    plt.plot(np.log(SGD_time), SGD_rel)
    plt.plot(np.log(Newton_time), Newton_rel)
    plt.xlabel('log(time)')
    plt.ylabel('relative value')
    plt.legend(['Pegasos', 'SGD', 'Newton'])
    plt.title('{}_Result: relative value over time'.format(DATA))
    plt.savefig('out/{}_Result_relative.png'.format(DATA))

    plt.figure()
    plt.plot(np.log(Pegasos_time), Pegasos_acc)
    plt.plot(np.log(SGD_time), SGD_acc)
    plt.plot(np.log(Newton_time), Newton_acc)
    plt.xlabel('log(time)')
    plt.ylabel('accuracy')
    plt.legend(['Pegasos', 'SGD', 'Newton'])
    plt.title('{}_Result: accuracy over time'.format(DATA))
    plt.savefig('out/{}_Result_accuracy.png'.format(DATA))





# Main entry point to the program
if __name__ == '__main__':
    main()
