import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cvxopt
import time
cvxopt.solvers.options['show_progress'] = False


def train_svm(train_data, train_label, C):
    """
    Argument:
      train_data: N*D matrix, each row as a sample and each column as a feature
      train_label: N*1 vector, each row as a label
      C: tradeoff parameter (on slack variable side)

    Return:
      w: feature vector (column vector)
      b: bias term
    """
    n = np.size(train_data, 0)
    sum_row = np.zeros(60)
    sum_b = 0
    for i in range(n):
        sum_row += train_label[i] * train_data[i, :]
        sum_b += train_label[i]
    sum_b = -C * sum_b
    Q = np.hstack((sum_row, sum_b))
    P = np.identity(61)
    P[60, 60] = 0
    Q = np.array(Q)
    P = np.mat(P)
    G = np.zeros((n, 61))
    for j in range(n):
        G[j, :] = np.mat(
            np.hstack((-1 * train_label[j] * train_data[j, :], [1])))
    Q = cvxopt.matrix(Q)
    P = cvxopt.matrix(P)
    G = cvxopt.matrix(G)
    h = np.zeros(n)
    h = cvxopt.matrix(h)
    sol = cvxopt.solvers.qp(P, Q, G, h)
    sol_array = sol['x']
    w = sol_array[:60]
    b = sol_array[60]
    return w, b


def test_svm(test_data, test_label, w, b):
    """
    Argument:
      test_data: M*D matrix, each row as a sample and each column as a feature
      test_label: M*1 vector, each row as a label
      w: feature vector
      b: bias term

    Return:
      test_accuracy: a float between [0, 1] representing the test accuracy
    """
    m = np.size(test_data, 0)
    count = 0
    for i in range(m):
        predict = (test_data[i, :].dot(w)) + b
        if predict >= 0:
            label1 = 1.0
        else:
            label1 = -1.0
        if label1 == test_label[i]:
            count += 1
        else:
            count += 0
    accuracy = count / np.size(test_label)
    return accuracy


train_data_path = "/Users/muzo01/Cpp_projects/hw4_program/train_data.txt"
train_label_path = "/Users/muzo01/Cpp_projects/hw4_program/train_label.txt"
test_data_path = "/Users/muzo01/Cpp_projects/hw4_program/test_data.txt"
test_label_path = "/Users/muzo01/Cpp_projects/hw4_program/test_label.txt"

train_data = np.genfromtxt(train_data_path)
train_label = np.genfromtxt(train_label_path)
test_data = np.genfromtxt(test_data_path)
test_label = np.genfromtxt(test_label_path)

sum_train = np.zeros(60)
sum_test = np.zeros(60)
ave_train = np.zeros(60)
ave_test = np.zeros(60)

for i in range(60):
    for j in range(1000):
        sum_train[i] += train_data[j, i]
ave_train = 1 / 1000 * sum_train
for k in range(60):
    for t in range(2175):
        sum_test[k] += test_data[t, k]
ave_test = 1 / 2175 * sum_test

s_train = np.zeros(60)
s_test = np.zeros(60)
squ_train = np.zeros(60)
squ_test = np.zeros(60)

for i in range(60):
    # sum up
    for j in range(1000):
        squ_train[i] += (train_data[j, i] - ave_train[i])**2
    # standard deviation for train feature i
    s_train[i] = (squ_train[i] / (1000 - 1))**(1/2)
for k in range(60):
    # sum up
    for j in range(2175):
        squ_test[k] += (test_data[j, k] - ave_test[k])**2
    # standard deviation for test feature i
    s_test[k] = (squ_test[k] / (2175 - 1))**(1/2)
print("3rd mean = ", ave_train[2])
print("3rd standard deviation = ", s_train[2])
print("10nd mean = ", ave_train[9])
print("10nd mean = ", s_train[9])
# w, b = train_svm(train_data, train_label, C)

# 5-fold cross validation
K = 5
for c in range(13):
    start = time.time()
    acc = np.zeros(5)
    C = 4 ** (c-6)
    for i in range(K):
        index = np.arange(200 * i, 200 * (i + 1))
        X_train = train_data
        X_train_label = train_label
        X_train = np.delete(X_train, index, 0)
        X_train_label = np.delete(X_train_label, index)
        X_test = train_data[200 * i: 200 * i + 200, :]
        X_test_label = train_label[200 * i: 200 * (i + 1)]
        w, b = train_svm(X_train, X_train_label, C)
        acc[i] = test_svm(X_test, X_test_label, w, b)
    acc_mean = np.sum(acc) / K
    end = time.time()
    print("C = 4 **", c-6, ", Accuracy =  ",
          acc_mean, ", Time = ", end - start)
"""
Increase C first decreased accuracy first and then increased,
Running time decreased first and then increased as C incresing.
Since C = 4^5 and 4^6 has the largest accuracy,
C = 4^6 has relatively shorter running time,
so choose C = 4^6.
"""
w, b = train_svm(train_data, train_label, 4**6)
accuracy = test_svm(test_data, test_label, w, b)
print("SVM test accuracy = ", accuracy)
