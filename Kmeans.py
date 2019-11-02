
from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(99)

means = [[2, 2], [8, 3], [3, 6]]
# độ tụ theo chiều ngang và chiều dọc
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T
print(X0)
# Draw multiple points.
def draw_multiple_points(X, pred_label):
    print((pred_label == 0).shape)
    print((pred_label == 0))
    print((pred_label == 1).shape)
    print((pred_label == 1))
    print((pred_label == 2).shape)
    print((pred_label == 2))
    X0 = X[pred_label == 0, 0]
    X1 = X[pred_label == 1, :]
    X2 = X[pred_label == 2, :]
    # X3 = X[pred_label == 3, :]
    # X4 = X[pred_label == 4, :]
    # Draw point based on above x, y axis values.
    # plt.scatter(X0[:, 0], X0[:, 1], s=10, c='red')
    # plt.scatter(X1[:, 0], X1[:, 1], s=10, c='blue')
    # plt.scatter(X2[:, 0], X2[:, 1], s=10, c='green')
    # plt.scatter(X3[:, 0], X3[:, 1], s=10, c='black')
    # plt.scatter(X4[:, 0], X4[:, 1], s=10, c='gray')

    # Draw only
    plt.scatter(X[:, 0], X[:, 1], s=10, c=pred_label)

    plt.scatter(center[:, 0], center[:, 1], s=500, c='yellow')
    plt.show()

# for index in range(len(x)):
#     color = ''
#     x = X[index]
#     label = pred_label[index]
#     if label == 0:
#         color == 'red'
#     if label == 1:
#         color == 'blue'
#     if label == 2:
#         color = 'green'
#     plt.scatter(center[:, 0], center[:, 1], s=10, c='color')
#     plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

pred_label = kmeans.predict(X)

print(pred_label)
print(pred_label[0])
print(pred_label.shape)

center = kmeans.cluster_centers_


draw_multiple_points(X, pred_label)