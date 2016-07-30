import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(np.random.seed(0))
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

print(['(knn.fit(iris_X_train, iris_y_train) ) \n'  ,knn.fit(iris_X_train, iris_y_train)] )
print(['(iris_X_test)                          \n',iris_X_test                         ])
print(['(knn.predict(iris_X_test))             \n' ,knn.predict(iris_X_test)            ])
print(['(iris_y_test)                          \n' ,iris_y_test                         ])                           
