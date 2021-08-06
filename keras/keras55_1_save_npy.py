from sklearn.datasets import load_iris,  load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100


datasets_iris = load_iris()

x_data_iris = datasets_iris.data
y_data_iris = datasets_iris.target

print(type(x_data_iris), type(y_data_iris))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('/study/_save/_npy/k55_x_data_iris.npy', arr = x_data_iris)
np.save('/study/_save/_npy/k55_y_data_iris.npy', arr = y_data_iris)


datasets_boston = load_boston()

x_data_boston = datasets_boston.data
y_data_boston = datasets_boston.target

print(type(x_data_boston), type(y_data_boston))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('/study/_save/_npy/k55_x_data_boston.npy', arr = x_data_boston)
np.save('/study/_save/_npy/k55_y_data_boston.npy', arr = y_data_boston)


datasets_cancer = load_breast_cancer()

x_data_cancer = datasets_cancer.data
y_data_cancer = datasets_cancer.target

print(type(x_data_cancer), type(y_data_cancer))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('/study/_save/_npy/k55_x_data_cancer.npy', arr = x_data_cancer)
np.save('/study/_save/_npy/k55_y_data_cancer.npy', arr = y_data_cancer)


datasets_diabet = load_diabetes()

x_data_diabet = datasets_diabet.data
y_data_diabet = datasets_diabet.target

print(type(x_data_diabet), type(y_data_diabet))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('/study/_save/_npy/k55_x_data_diabet.npy', arr = x_data_diabet)
np.save('/study/_save/_npy/k55_y_data_diabet.npy', arr = y_data_diabet)


datasets_wine = load_wine()

x_data_wine = datasets_wine.data
y_data_wine = datasets_wine.target

print(type(x_data_wine), type(y_data_wine))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('/study/_save/_npy/k55_x_data_wine.npy', arr = x_data_wine)
np.save('/study/_save/_npy/k55_y_data_wine.npy', arr = y_data_wine)