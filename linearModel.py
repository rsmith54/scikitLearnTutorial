import numpy as np
from sklearn import datasets

diabetes = datasets.load_diabetes()

diabetes = {
    'test'  : (diabetes.data[-20:], diabetes.target[-20:] ),
    'train' : (diabetes.data[:-20] , diabetes.target[:-20]),
}

from sklearn import linear_model
regr = linear_model.LinearRegression()


print ( regr.fit(diabetes['train'][0], diabetes['train'][1]))
print(regr.coef_)
print(np.mean((regr.predict(diabetes['test'][0])-diabetes['test'][1])**2))

print(regr.score(diabetes['test'][0], diabetes['test'][1]) )
