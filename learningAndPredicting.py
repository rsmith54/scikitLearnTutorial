from sklearn import datasets
from sklearn import svm

iris   = datasets.load_iris()
digits = datasets.load_digits()


#print(iris.data)
#print(digits.data)

#print(iris.target)
#print(digits.target)

clf = svm.SVC(gamma=0.001, C=100., verbose = True)
print(clf)

clf.fit(digits.data[:-1], digits.target[:-1])
#fit using all but the last data

clf.predict(digits.data[-1:])
#predit the classification of the final element of digits

print()

print(clf.predict(makeIntoSix))
