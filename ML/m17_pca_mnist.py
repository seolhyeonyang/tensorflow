import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


(x_train, _), (x_test, _) = mnist.load_data()
#! 빈칸으로 한다는뜻, 따로 변수로 안 받는다는 이야기다.

print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape)      # (70000, 28, 28)

x = x.reshape(70000, 28 * 28)

pca = PCA(n_components=28 * 28)
#! 3차원은 안되서 reshapme로 2차원으로 만들고 사용

x = pca.fit_transform(x)
print(x)
print(x.shape)     # (442, 7)


pca_EVR = pca.explained_variance_ratio_

# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)     # 누적합 구하는 것
# print(cumsum)

print(np.argmax(cumsum >=0.95)+1)