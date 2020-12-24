import pandas as pd
import numpy as np
import random
import math
import argparse
import time

def sigmoid(iX, dimension):  # iX is a matrix with a dimension
    if dimension == 1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = sigmoid(iX[i], dimension - 1)
    return iX


def mean_error(epoch_error):
    neg_error = np.mean(np.abs(np.array(epoch_error[:8]) - np.array([1])))
    pos_error = np.mean(np.abs(np.array(epoch_error[8:]) - np.array([0])))
    return np.mean(neg_error + pos_error)

# standard BP
def std_bp(X, trueY, maxIter, gamma, theta, eta, w, v, m, q, l, d):
    error_list = []
    while (maxIter > 0):
        maxIter -= 1
        sumE = 0
        # epoch_error = []
        for i in range(m):
            alpha = np.dot(X[i], v)  # p101 line 2 from bottom, shape=1*q
            b = sigmoid(alpha - gamma, 1)  # b=f(alpha-gamma), shape=1*q
            beta = np.dot(b, w)  # shape=(1*q)*(q*l)=1*l
            predictY = sigmoid(beta - theta, 1)  # shape=1*l ,p102--5.3
            # epoch_error.append(predictY)
            E = sum((predictY - trueY[i]) * (predictY - trueY[i])) / 2  # 5.4
            sumE += E  # 5.16
            g = predictY * (1 - predictY) * (trueY[i] - predictY)  # shape=1*l p103--5.10
            e = b * (1 - b) * ((np.dot(w, g.T)).T)  # shape=1*q , p104--5.15
            w += eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))  # 5.11
            theta -= eta * g  # 5.12
            v += eta * np.dot(X[i].reshape((d, 1)), e.reshape((1, q)))  # 5.13
            gamma -= eta * e  # 5.14

        # error_list.append(mean_error(epoch_error))
    return v, w, b, gamma ,theta, error_list

# #accumulated BP
def accumulated_bp(X, trueY, maxIter, gamma, theta, eta, w, v, m, q, l, d):
    trueY = trueY.reshape((m, l))
    error_list = []
    while maxIter > 0:
        maxIter -= 1
        alpha = np.dot(X, v)  # p101 line 2 from bottom, shape=m*q
        b = sigmoid(alpha - gamma, 2)  # b=f(alpha-gamma), shape=m*q
        beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
        predictY = sigmoid(beta - theta, 2)  # shape=m*l ,p102--5.3
        # epoch_error = predictY
        E = sum(sum((predictY - trueY) * (predictY - trueY))) / 2  # 5.4
        g = predictY * (1 - predictY) * (trueY - predictY)  # shape=m*l p103--5.10
        e = b * (1 - b) * ((np.dot(w, g.T)).T)  # shape=m*q , p104--5.15
        w += eta * np.dot(b.T, g)  # 5.11 shape (q*l)=(q*m) * (m*l)
        theta -= eta * g  # 5.12
        v += eta * np.dot(X.T, e)  # 5.13 (d,q)=(d,m)*(m,q)
        gamma -= eta * e  # 5.14
        # error_list.append(mean_error(epoch_error))
    return v, w, b, gamma ,theta, error_list

def predict(iX, gamma, theta,  w, v):

    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid(alpha - gamma, 2)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid(beta - theta, 2)  # shape=m*l ,p102--5.3
    return predictY

def main():
    print('Model type:', model_type, ' lr:', eta,' maxIter:',maxIter)
    dataset = pd.read_csv(data_dir, delimiter=",")
    del dataset['编号']
    dataset = np.array(dataset)
    m, n = np.shape(dataset)

    attributeMap = {'浅白': 0, '青绿': 0.5, '乌黑': 1, '蜷缩': 0, '稍蜷': 0.5, '硬挺': 1, '沉闷': 0, '浊响': 0.5, '清脆': 1,
                    '模糊': 0, '稍糊': 0.5, '清晰': 1, '凹陷': 0, '稍凹': 0.5, '平坦': 1, '硬滑': 0, '软粘': 1, '否': 0, '是': 1}

    for i in range(m):
        for j in range(n):
            if dataset[i, j] in attributeMap:
                dataset[i, j] = attributeMap[dataset[i, j]]
            dataset[i, j] = round(dataset[i, j], 3)

    trueY = dataset[:, n - 1]
    X = dataset[:, :n - 1]
    m, n = np.shape(X)
    d = n
    l = 1
    q = d + 1
    theta = [random.random() for i in range(l)]
    gamma = [random.random() for i in range(q)]
    v = [[random.random() for i in range(q)] for j in range(d)]
    w = [[random.random() for i in range(l)] for j in range(q)]

    if model_type == 'standard':
        time_begin = time.time()
        v, w, b, gamma, theta, error_list = std_bp(X, trueY, maxIter, gamma, theta, eta, w, v, m, q, l, d)
        time_end = time.time()
    elif model_type == 'accumulate':
        time_begin = time.time()
        v, w, b, gamma, theta, error_list = accumulated_bp(X, trueY, maxIter, gamma, theta, eta, w, v, m, q, l, d)
        time_end = time.time()
    else:
        raise AssertionError
    print('time: ', time_end - time_begin)
    print(predict(X, gamma, theta,  w, v))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./watermelon_3.csv')
    parser.add_argument('--eta', type=float, default = 0.2)
    parser.add_argument('--maxIter', type=int, default=50000)
    parser.add_argument('--model_type', type=str, default='standard')
    args = parser.parse_args()

    data_dir = args.data_dir
    eta = args.eta
    maxIter = args.maxIter
    model_type = args.model_type

    main()



