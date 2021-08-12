import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# 1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

pca = PCA(n_components=7)
#! 컬럼이나 차원 축소 해주는것 ex) 컬럼 10개를 7개로 축소해 준다.

x = pca.fit_transform(x)
print(x)
print(x.shape)     # (442, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# 2. 모델
from xgboost import XGBRFRegressor
model = XGBRFRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 : ', results)


'''
xgb 결과 :  0.36759127357720334
pca 결과 :  0.3723109192112878
'''