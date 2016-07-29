import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
print(X)
Xarray = np.array(X, dtype='float32')
print(Xarray)
print(Xarray.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(Xarray)
print(X_new.dtype)
