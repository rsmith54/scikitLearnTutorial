from sklearn import cross_validation
from sklearn import datasets, svm

import copy
import numpy as np
from sklearn import cross_validation, datasets, svm

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = []
for c in C_s :
    svc_C = copy.deepcopy(svc)
    svc_C.C = c
    scores.append(cross_validation.cross_val_score( svc_C, X, y , # cv = cross_validation.KFold(len(X), n_folds = 10) ,
                                                    n_jobs = 4) )

import matplotlib.pyplot as plt

plt.semilogx(C_s, scores)

plt.xlabel('c value')
plt.ylabel('score  ')
plt.grid(True)
plt.show()

# digits = datasets.load_digits()
# X_digits = digits.data
# y_digits = digits.target
# svc = svm.SVC(C=1, kernel='linear')

# print(svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:]))

# import numpy as np
# X_folds = np.array_split(X_digits, 3)
# y_folds = np.array_split(y_digits, 3)
# scores = list()
# for k in range(3):
#     # We use 'list' to copy, in order to 'pop' later on
#     X_train = list(X_folds)
#     X_test  = X_train.pop(k)
#     X_train = np.concatenate(X_train)
#     y_train = list(y_folds)
#     y_test  = y_train.pop(k)
#     y_train = np.concatenate(y_train)
#     scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

# print(scores)

# kfold = cross_validation.KFold(len(X_digits), n_folds=3)
# print([svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in kfold])
# print(cross_validation.cross_val_score(svc, X_digits, y_digits, cv=kfold, n_jobs=-1))
