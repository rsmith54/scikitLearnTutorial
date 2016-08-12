from sklearn import linear_model
from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

#import the lasso
regr = linear_model.Lasso()

alphas = np.logspace(-4, -1, 6)

scores = [regr.set_params(alpha=alpha
                          ).fit(diabetes_X_train, diabetes_y_train
                          ).score(diabetes_X_test, diabetes_y_test)
          for alpha in alphas]

print(scores)

print(alphas)

best_alpha = alphas[scores.index(max(scores))]
print(best_alpha)
regr.alpha = best_alpha

regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
