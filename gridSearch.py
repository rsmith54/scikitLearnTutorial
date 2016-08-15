
from sklearn import datasets,svm
from sklearn.grid_search import GridSearchCV
import numpy as np

from sklearn import cross_validation

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc,
                   param_grid=dict(C=Cs),
                   n_jobs=4)
clf.fit(X_digits[:1000], y_digits[:1000])

print(clf.best_score_)

print(clf.best_estimator_.C)

# Prediction performance on test set is not as good as on train set
print(clf.score(X_digits[1000:], y_digits[1000:]))

print(cross_validation.cross_val_score(clf, X_digits, y_digits))
