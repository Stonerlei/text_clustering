# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:19:50 2020

@author: Stoner
"""

import os
import numpy as np
from data_divide import split_train_test
from sklearn.feature_extraction.text import CountVectorizer
import random
from k_means import k_mean
from sklearn.model_selection import KFold
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


# preprocessing
path = "C:\\Users\\Stoner\\Desktop\\大二下课程\\NLP-PPT\\text_classification\\20_newsgroups"
class_names = os.listdir(path)
path = path + "\\"
# print(class_names)
train_set = []
test_set = []
for class_name in class_names:
    file_path = path + class_name
    file_names = os.listdir(file_path)
    file_path = file_path + "\\"
    files = []
    for file_name in file_names:
        with open(file_path + file_name, errors="ignore") as file:
            A = file.read()
            files.append(A)
    files = np.asarray(files)
    train_set.append(files)
# print(files[0])
# print(len(files))
# print(len(train_set))
# print(train_set)

# construct the frequency matrix
# stop words
stopdir = "C:\\Users\\Stoner\\Desktop\\大二下课程\\NLP-PPT\\text_classification\\english.txt"
with open(stopdir, errors='ignore') as stopdir:
    stopword = stopdir.read()
stopword = stopword.splitlines()
stopword = stopword + ['ain', 'daren', 'hadn', 'mayn', 'mightn', 'mon', 'mustn', 'needn', 'oughtn', 'shan']
# 将训练集合并成易操作的格式
seq_train = []
# print(train_set[0][0])
for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        seq_train.append(train_set[i][j])
# 构建词袋模型
cv_train = CountVectorizer(stop_words=stopword, lowercase=True, max_features=50000)
cv_train_fit = cv_train.fit_transform(seq_train)
freq_matrix_train = cv_train_fit.toarray()


num_center = 20
center_ind = []
center = []
for i in range(num_center):
    center.append(freq_matrix_train[random.randint(0,19999)])

# knn
freq_matrix_train = np.asarray(freq_matrix_train)

# test if func works well
# y_pred = np.zeros(len(freq_matrix_train))
# new_center = np.zeros(np.shape(center))
# count = np.zeros(num_center)
# for j in range(len(freq_matrix_train)):
#     distance = np.zeros(num_center)
#     for k in range(num_center):
#         product = freq_matrix_train[j] * center[k]
#         distance[k] = np.sum(product)
#     y_pred[j] = np.argmin(distance)
#     new_center[int(y_pred[j])] += freq_matrix_train[j]
#     count[int(y_pred[j])] += 1
# for k in range(num_center):
#     new_center[k] /= count[k]

# print(np.shape(freq_matrix_train))
[center, count, y_pred, SSE] = k_mean(freq_matrix_train, center, num_center)

num_iteration = 2000
for k in range(num_iteration-1):
    temp_SSE = SSE
    [center, count, y_pred, SSE] = k_mean(freq_matrix_train, center, num_center)
    print('第',k,'次迭代：')
    print('目标函数为本次迭代前的',SSE/temp_SSE,'倍')
    if SSE/temp_SSE > 0.999999:
        break


y_true = []
for j in range(len(freq_matrix_train)):
    y_true.append(int(j/1000))
ARI = adjusted_rand_score(y_true,y_pred)
NMI = normalized_mutual_info_score(y_true,y_pred)

majority = np.zeros([num_center,num_center], dtype=int)
for j in range(len(freq_matrix_train)):
    for k in range(num_center):
        if y_pred[j] == k:
            majority[k][int(j/1000)] += 1
correct = 0
for k in range(num_center):
    correct += np.max(majority[k])/len(freq_matrix_train)
    
for j in range(len(freq_matrix_train)):
    for k in range(num_center):
        if y_pred[j] == k:
            y_pred[j] = np.argmax(majority[k])
      

confusion = np.zeros([20,20])
for j in range(len(freq_matrix_train)):
    confusion[int(j/1000)][int(y_pred[j])] += 1
