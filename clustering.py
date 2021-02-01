import os
import numpy as np
from data_divide import split_train_test
from sklearn.feature_extraction.text import CountVectorizer
import random
from k_means import k_mean
from sklearn.model_selection import KFold

# preprocessing
path = "C:\\Users\\Stoner\\Desktop\\大二下课程\\NLP-PPT\\text_classification\\mini_newsgroups"
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
cv_train = CountVectorizer(stop_words=stopword, lowercase=True, max_features=5000)
cv_train_fit = cv_train.fit_transform(seq_train)
# print(len(seq_train))
# print(cv.vocabulary_)
freq_matrix_train = cv_train_fit.toarray()
# print(freq_matrix_train)
# print(len(cv_train.get_feature_names()))


num_center = 20
center_ind = []
center = []
k_value = 2000
# for i in range(num_center):
#     center_ind.append(random.randint(0, k_value * num_center))
#     center.append(freq_matrix_train[i])
for i in range(num_center):
    center.append(freq_matrix_train[100*i+20])

# knn
freq_matrix_train = np.asarray(freq_matrix_train)
# print(np.shape(freq_matrix_train))
k_max = k_mean(freq_matrix_train, center, num_center, k_value)
# print(np.shape(k_max))
# k_max = np.zeros((num_center, 2, k_value))
# for k in range(num_center):
#     for j in range(len(freq_matrix_train)):
#         distance = freq_matrix_train[j] * center[k]
#         distance = np.sum(distance)
#         # print('distance: ', distance)
#         # print(np.argmin(k_max[0]))
#         if distance > np.min(k_max[k][0]):
#             # print('yes', np.argmin(k_max[0]))
#             # print('distance: ', distance, 'j: ', j)
#             # print('k_max: ', k_max)
#             k_max[k, 1, np.argmin(k_max[0])] = j  # 与训练集大小有关
#             k_max[k, 0, np.argmin(k_max[0])] = distance

num_iteration = 20
for k in range(num_iteration-1):
    for i in range(num_center):
        center[i] = np.zeros(np.shape(center[i]))  # shape使用可能有错
        for j in range(k_value):
            center[i] = center[i] + freq_matrix_train[int(k_max[i][1][j])]
        center[i] = center[i] / k_value
    k_max = k_mean(freq_matrix_train, center, num_center, k_value)
    print(k)
correct_rate = np.zeros(num_center)
for i in range(num_center):
    majority = np.zeros((1, num_center))
    for j in range(num_center):
        for k in range(k_value):
            if int(k_max[i][1][k] / k_value) == j:
                majority[0, j] = majority[0, j] + 1
    correct_rate[i] = np.max(majority)

print('Correct Rate: ', correct_rate/k_value)
