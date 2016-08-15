#Exercise
#Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features.
#Leave out 10% of each class and test prediction performance on these observations.
#Warning: the classes are ordered, do not leave out the last 10%, you would be testing on only one class.
#Hint: You can use the decision_function method on a grid to get intuitions.

from sklearn import datasets, svm
import numpy as np



iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

np.random.seed(0)
indices = np.random.permutation(len(X))

iris_X_train = X[indices[:-10]]
iris_y_train = y[indices[:-10]]
iris_X_test  = X[indices[-10:]]
iris_y_test  = y[indices[-10:]]

C = 1.0  # SVM regularization parameter
svcDict      = {'linear' : svm.SVC      (kernel='linear', gamma=10),
                'rbf'    : svm.SVC      (kernel='rbf'   , gamma=10),
                'poly3'  : svm.SVC      (kernel='poly'  , gamma=10),
            }
#lin_svc  = svm.LinearSVC(C=C)                 

for name , svc in svcDict.items() :
    svc.fit(iris_X_train, iris_y_train)

    print(svc)
    print( [name + 'predict : ' , svc.predict(iris_X_test)])
    print( ["score : " ,    svc.fit(iris_X_train, iris_y_train).score(iris_X_test,iris_y_test) ])
    #    print([name + ' predict : ' , svc.predict(X_test)])

    print('\n')
