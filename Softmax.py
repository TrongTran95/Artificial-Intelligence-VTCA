import numpy as np
from sklearn.model_selection import train_test_split


# Generate random points surround 3 centroids: [2,2], [8,3], [3,6]
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 750
C = 3

X_0 = np.random.multivariate_normal(means[0], cov, N)
X_1 = np.random.multivariate_normal(means[1], cov, N)
X_2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X_0, X_1, X_2), axis = 0).T


# extended data: append a colum of number 1 to the left
X = np.concatenate((np.ones((1, C*N)), X), axis = 0).T
y = np.array([[0]*N,[1]*N,[2]*N]).reshape(C*N,)

# split train set and test set from data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print("---------X_train--------")
print(X_train)
print("---------X_test--------")
print(X_test)
print("---------y_train--------")
print(y_train)
print("---------y_test--------")
print(y_test)



def draw_multiple_points(X, pred_label):
    print((pred_label == 0).shape)
    print((pred_label == 0))
    print((pred_label == 1).shape)
    print((pred_label == 1))
    print((pred_label == 2).shape)
    print((pred_label == 2))
    X0 = X[pred_label == 0, :]
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




from sklearn import linear_model
from sklearn.metrics import accuracy_score

# For a multi_class problem, if multi_class is set to
# be “multinomial” the softmax function is used to
# find the predicted probability of each class
logreg = linear_model.LogisticRegression(C=1e5,
        solver = 'lbfgs', multi_class = 'multinomial')

# train
logreg.fit(X_train, y_train)
# test
y_pred = logreg.predict(X_test)
print("---------y_pred--------")
print(y_pred)

#evaluate
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred.tolist())))