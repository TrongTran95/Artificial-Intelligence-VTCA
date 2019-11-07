import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model

#binary classification

def sign(x):
    # TODO: implement sign function. Sign function
    #       returns 1  if x is positive,
    #       returns -1 otherwise
    pass

# them mot cot 1 truoc moi X
def generate_X_dot(X):
    # TODO: Append a column of number 1 to the left of X
    print(X.shape)
    return (np.ones((1, C*N)), X)

# su dung cong thuc predict voi mot cai X dua vao, predict voi tat ca X_dot
def predict_single_data(x_dot, w):
    # TODO: implement function to predict label of x_dot
    y = np.dot(w, x_dot)
    S = sign(y)
    return S


def predict(X_dot, w):
    # TODO: calculate y from data X_dot and weight w
    pass


def update_w(x_dot, w, y, learning_rate):
    # TODO: implement function to update w
    # w_new = w + learning_rate *
    pass

# predict voi moi x cho ra y, xet cai y predict voi moi label cua x do, neu y_pred khac voi label cua x do, thi ...
def train(X, y, epochs, learning_rate):
    w = np.zeros(len(X[0]))

    print("-----f(train), w-----")
    print(w)
    # TODO: generate X_dot
    X_dot = generate_X_dot(X)
    print(X_dot[0].shape)
    print(X_dot[0])
    print(X_dot[1].shape)
    print(X_dot[1])
    # for epoch in range(epochs):
    #     print(epoch)
        # print(X_dot[epoch])
    # TODO:
    # - predict label of every point x_i
    # - if this point is missclassified, update w

    return w

# Generate data set

N = None
means = [[2, 2], [4, 4]]
cov = [[1, 0], [0, 1]]

N = 10
C = 2

X_0 = np.random.multivariate_normal(means[0], cov, N)
X_1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X_0, X_1), axis = 0)
y = np.array([[0]*N,[1]*N]).reshape(C*N,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

print("-----X_train-----")
print(X_train)
print("-----X_test-----")
print(X_test)
print("-----y_train-----")
print(y_train)
print("-----y_test-----")
print(y_test)

w = train(X_train, y_train, epochs = 100, learning_rate=0.01)

# TODO: calculate y_pred

# print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# TODO: Visulize X_test on 2D graph, colored by y_test

# TODO: Visualize X_test on 2D graph, colored by y_pred

def train_verbose(X, y, epochs, learning_rate):
  w = np.zeros(len(X[0]))
  error_history = []

  # TODO: implement training algorithm again and
  #       remember to append error to error_history
  #       in every epoch

  return w, error_history

w, error_history = train_verbose(X_train, y_train, learning_rate=0.1, epochs = 20)

# TODO: visualize error_history by the graph

ones = np.ones(X.shape[0], order = 1)
