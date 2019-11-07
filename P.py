# generate data
# list of points
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
# Xbar
X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)


def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)  # True if h(w, X) == y else False


def perceptron(X, y, w_init):
    w = [w_init]
    print("\nX")
    print(X)
    print("\nX[0]")
    print(X[0])
    print("\nX[1]")
    print(X[1])
    print("\ny")
    print(y)
    print("\nw_init")
    print(w_init)
    print("\nw")
    print(w)
    print("\nX.shape[1]")
    print(X.shape[1])
    N = X.shape[1]
    print("\nN")
    print(N)
    mis_points = []
    while True:
        # mix data
        mix_id = np.random.permutation(N)
        print("\n\nmix_id")
        print(mix_id)
        for i in range(N):
            # print(X[:, mix_id[i]])
            xi = X[:, mix_id[i]].reshape(3, 1)
            print("xi")
            print(xi)
            yi = y[0, mix_id[i]]
            print("yi")
            print(yi)
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi

                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)


d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)

# print(m)

