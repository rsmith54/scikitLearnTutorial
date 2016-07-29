import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()

print(iris.data.shape)

digits = datasets.load_digits()
print(digits.images.shape)
#import matplotlib as mpl


plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)


import time
time.sleep(1000)
