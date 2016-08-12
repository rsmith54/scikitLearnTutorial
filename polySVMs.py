print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
digits = datasets.load_digits()
X = digits.data[:-10]  # we only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset
y = digits.target[:-10]

X_test = digits.data  [-10:]
y_test = digits.target[-10:]

C = 1.0  # SVM regularization parameter
svcDict      = {'linear' : svm.SVC      (kernel='linear', C=C),
                'rbf'    : svm.SVC      (kernel='rbf'   , C=C),
                'poly3'  : svm.SVC      (kernel='poly', degree=3, C=C),
            }
#lin_svc  = svm.LinearSVC(C=C)                 

for name , svc in svcDict.items() :
    svc.fit(X, y)
    
    print([name + ' predict : ' , svc.predict(X_test)])
    print([name + ' score   : '])
    print(svc.fit(X,y).score(X_test, y_test))

    print('\n')

# print(['lin_svc predict : ' , lin_svc.predict(X_test)])
# print(['lin_svc score   : '])
# print([lin_svc.fit(X,y).score(X_test, y_test)])

