#Exercise
#On the diabetes dataset, find the optimal regularization parameter alpha.
#Bonus: How much can you trust the selection of alpha?

from sklearn import cross_validation, datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)
