import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# 1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

pca = PCA(n_components=10)

x = pca.fit_transform(x)
print(x)
print(x.shape)     # (442, 7)


pca_EVR = pca.explained_variance_ratio_

print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)     # 누적합 구하는 것
print(cumsum)
#! EVR를 확인해서 몇개의 컬럼을 사용할지 정한다. (누적합을 확인해 기준치에 해당하는것을 확인한다.)

'''
n_components=10
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
0.05365605 0.04336832 0.00783199 0.00085605]
1.0

n_components=9
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
0.05365605 0.04336832 0.00783199]
0.9991439470098977

n_components=7
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
0.05365605]
0.9479436357350414

#! 기여가 낮은 컬럼부터 줄여 준다.
'''

print(np.argmax(cumsum >=0.94)+1)
#^ 누적합 0.94 이상인 개수에 +1

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# 2. 모델
from xgboost import XGBRFRegressor
model = XGBRFRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 : ', results)

# 결과 :  0.3616983428659313