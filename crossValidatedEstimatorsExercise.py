#Exercise
#On the diabetes dataset, find the optimal regularization parameter alpha.
#Bonus: How much can you trust the selection of alpha?

from sklearn import cross_validation, datasets, linear_model
import copy
import numpy as np

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

X_test = diabetes.data[150:]
y_test = diabetes.target[150:]


lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

scores = []

for alpha in alphas : 
    print(alpha)
    lasso_alpha = copy.deepcopy(lasso)
    lasso_alpha.alpha = alpha

    scores.append(lasso_alpha.fit(X,y).score(X_test, y_test))

import matplotlib.pyplot as plt

plt.plot(alphas, scores)
plt.xlabel('alpha  ')
plt.ylabel('score  ')
plt.grid(True)

plt.show()
