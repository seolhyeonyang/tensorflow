from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성

# model = KNeighborsRegressor()
# Acc : [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 0.3673

# model = LinearRegression()
# Acc : [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 0.4876

# model = DecisionTreeRegressor()
# Acc : [-0.25314398 -0.12196638 -0.19540696 -0.02005442  0.02050502] -0.114

model = RandomForestRegressor()
# Acc : [0.36850405 0.50690339 0.47894478 0.37479161 0.43024097] 0.4319

# 3. 컴파일, 훈련
# 4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)

print('Acc :', scores, round(np.mean(scores),4))