#Exercise
#Try classifying the digits dataset with nearest neighbors and a linear model.
#Leave out the last 10% and test prediction performance on these observations.

from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

digits_X_train = digits_X[indices[:-10]]
digits_y_train = digits_y[indices[:-10]]
digits_X_test  = digits_X[indices[-10:]]
digits_y_test  = digits_y[indices[-10:]]

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)

print(logistic)
