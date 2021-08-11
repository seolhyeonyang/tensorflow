from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 2. 모델 구성
# model = KNeighborsRegressor()
# Acc : [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 0.5286

# model = LinearRegression()
# Acc : [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 0.7128

# model = DecisionTreeRegressor()
# Acc : [0.82082843 0.75921242 0.80305605 0.73788978 0.82745644] 0.7897

model = RandomForestRegressor()
# Acc : [0.92616366 0.84576502 0.81931826 0.88376577 0.90401632] 0.8758

# 3. 컴파일, 훈련
# 4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)

print('Acc :', scores, round(np.mean(scores),4))