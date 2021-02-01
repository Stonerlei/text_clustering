import numpy as np
import math

def k_mean(freq_matrix_train, center, num_center):
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
    #             k_max[k, 1, np.argmin(k_max[k][0])] = j  # 与训练集大小有关
    #             k_max[k, 0, np.argmin(k_max[k][0])] = distance
    SSE = 0
    y_predict = np.zeros(len(freq_matrix_train))
    new_center = np.zeros(np.shape(center))
    count = np.zeros(num_center)
    for j in range(len(freq_matrix_train)):
        distance = np.zeros(num_center)
        for k in range(num_center):
            product = freq_matrix_train[j] * center[k]
            distance[k] = 1 - np.sum(product)/math.pow(np.sum(freq_matrix_train[j]*freq_matrix_train[j]),0.5)/math.pow(np.sum(center[k]*center[k]),0.5)
            # if math.isnan(distance[k]):
            #     print('nan!!!')
            # dif = freq_matrix_train[j] - center[k]
            # distance[k] = math.pow(np.sum(dif*dif),0.5)
        y_predict[j] = np.argmin(distance)
        SSE += math.pow(distance[int(y_predict[j])],2)
        new_center[int(y_predict[j])] += freq_matrix_train[j]
        count[int(y_predict[j])] += 1
    for k in range(num_center):
        if count[k] == 0:
            pass
        else:
            new_center[k] /= count[k]
    return new_center, count, y_predict,SSE

# adwa = []
# for i in range(20):
#     distannn = 1 - np.sum(center[i]*freq_matrix_train[5050])/math.pow(np.sum(freq_matrix_train[5050]*freq_matrix_train[5050]),0.5)/math.pow(np.sum(center[i]*center[i]),0.5)
#     adwa.append(distannn)
#     np.argmin(adwa)
