import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model

#binary classification

def sign(x):
    # x is a scalar (number)
    # TODO: implement sign function. Sign function
    #       returns 1  if x is positive,
    #       returns -1 otherwise
    if x > 0:
        return 1
    else:
        return -1

def generate_X_dot(X):
    # X is a matrix size n*f, where:
    #   n is the number of sample x in X
    #   f is the number of features of a single sample x
    # TODO: Append a column of number 1 to the left of
    # print("aaaaaaaa-----------aaaaaaaaa")
    # print(X)
    a = np.ones((2, 1))
    # print(a)
    # print(np.ones((1, 2)))
    X_dot = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
    # print("aaaaaaaa-----------aaaaaaaaa")
    # print("aaaaaaaa-----------aaaaaaaaa")
    #
    # print(X_dot)
    return X_dot

# su dung cong thuc predict voi mot cai X dua vao, predict voi tat ca X_dot
def predict_single_data(x_dot, w_dot):
    # x_dot is an array contains f+1 element, where
    #       f is the number of feature of x
    # w_dot is a column vector that has the same size with x_dot
    # TODO: implement function to predict label of x_dot
    return sign(np.dot(w_dot, x_dot))

def predict(X_dot, w_dot):
    # X_dot is a matrix size n*(f+1), where:
    #       n is the number of sample x_dot in X_dot
    #       f is the number of feature of x in X
    # TODO: calculate y from data X_dot and weight w_dot
    y_pred = []
    # print("X_dot.shape[0]")
    # print(X_dot.shape[0])
    for index in range(X_dot.shape[0]):
        y_pred.append(predict_single_data(X_dot[index], w_dot))
    return np.array(y_pred)

def update_w_dot(x_dot, w_dot, y, learning_rate):
    # x_dot is an array contains f+1 element, where
    #       f is the number of feature of x
    # w_dot is a column vector that has the same size with x_dot
    # y     is the corresponding label of current x_dot
    # learning_rate is a float
    # TODO: implement function to update w_dot
    # print("x_dot.shape")
    # print(x_dot.shape)
    # print("w_dot.shape")
    # print(w_dot.shape)
    # print("w_dot.shape")
    # print(w_dot.shape)
    return w_dot + learning_rate * x_dot * y


# predict voi moi x cho ra y, xet cai y predict voi moi label cua x do, neu y_pred khac voi label cua x do, thi ...
def train(X, y, epochs, learning_rate):
    # print(y)
    # print("X.shape")
    # print(X.shape)
    # # TODO: generate X_dot
    X_dot = generate_X_dot(X)
    # print("X_dotX.shape")
    # print(X_dot.shape)
    w_dot = np.zeros(len(X_dot[0]))
    # print("X_dot[1].shaape")
    # print(X_dot[0].shape)
    # print(X_dot[0])
    # print(X_dot[1].shape)
    # print(X_dot[1])
    # print("range(epochs)")
    # print((epochs))
    # print(range(epochs))
    count = []
    for epoch in range(epochs):
        counter = 0
        for index in range(X_dot.shape[0]):
            x_dot = X_dot[index]
            # print()
            # print("x_dot")
            # print(x_dot)
            # print("w_dot")
            # print(w_dot)
            if (predict_single_data(x_dot, w_dot) != y[index]):
                w_dot = update_w_dot(x_dot, w_dot, y[index], learning_rate)
                counter += 1
        count.append(counter)
    # TODO:
    # - predict label of every point x_i
    # - if this point is missclassified, update w_dot

    return w_dot, count

# Generate data set

N = None
means = [[2, 2], [4, 4]]
cov = [[1, 0], [0, 1]]

N = 500
C = 2

X_0 = np.random.multivariate_normal(means[0], cov, N)
X_1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X_0, X_1), axis = 0)
y = np.array([[-1]*N,[1]*N]).reshape(C*N,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

print("-----X_train-----")
print(X_train)
print("-----X_test-----")
print(X_test)
print("-----y_train-----")
print(y_train)
print("-----y_test-----")
print(y_test)

w_dot, error_his = train(X_train, y_train, epochs = 100, learning_rate=0.01)

print("-----X_test----------X_test----------X_test----------X_test----------X_test-----")
print(error_his)
X_test = generate_X_dot(X_test)
print("-----X_test-----")
print(X_test.shape)
print("-----w_dot-----")
print(w_dot.shape)
y_pred = predict(X_test, w_dot)
print(y_test)
print(y_test.shape)
print(y_pred)
print(y_pred[0])
#
# # TODO: calculate y_pred
#
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
#

def ShowDiagram():
    # X_test0 = X_test[y_test == 0, :]
    # X_test1 = X_test[y_test == 1, :]
    # X_test2 = X_test[y_test == 2, :]
    # plt.scatter(X_test0[:, 1], X_test0[:, 2], s=10, c='red')
    # plt.scatter(X_test1[:, 1], X_test1[:, 2], s=10, c='blue')
    # plt.scatter(X_test2[:, 1], X_test2[:, 2], s=10, c='green')

    # X_test0 = X_test[y_test == 1, :]
    # X_test1 = X_test[y_test == -1, :]
    # plt.scatter(X_test0[:, 1], X_test0[:, 2], s=10, c='red')
    # plt.scatter(X_test1[:, 1], X_test1[:, 2], s=10, c='blue')

    # X_y_test0 = X_test[y_pred == 1, :]
    # X_y_test1 = X_test[y_pred == -1, :]
    # plt.scatter(X_y_test0[:, 1], X_y_test0[:, 2], s=10, c='red')
    # plt.scatter(X_y_test1[:, 1], X_y_test1[:, 2], s=10, c='blue')
    plt.plot(range(len(error_his)), error_his)
    plt.show()


ShowDiagram()
