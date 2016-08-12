#Exercise
#Try classifying the digits dataset with nearest neighbors and a linear model.
#Leave out the last 10% and test prediction performance on these observations.

from sklearn import datasets, neighbors, linear_model
import numpy as np

digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target

np.random.seed(0)
indices = np.random.permutation(len(digits_X))

digits_X_train = digits_X[:.9 * len(digits_X)]
digits_y_train = digits_y[:.9 * len(digits_X)]
digits_X_test  = digits_X[.9 * len(digits_X):]
digits_y_test  = digits_y[.9 * len(digits_X):]

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(digits_X_train, digits_y_train)

print(logistic)
print([logistic.predict(digits_X_test)])
print(digits_y_test)

print('LogisticRegression score: %f' % logistic.fit(digits_X_train, digits_y_train).score(digits_X_test, digits_y_test))
