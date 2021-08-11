from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parmeters = [
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

parmeters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [5, 7, 10]},
    {'n_jobs' : [-1], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [3, 6, 9, 11], 'min_samples_split' : [3, 4, 5]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7], 'min_samples_split' : [3, 4, 5]},
    {'n_jobs' : [-1], 'min_samples_split' : [2, 3, 5, 10]}
]

# 2. 모델 구성

model = GridSearchCV(RandomForestRegressor(), parmeters, cv=kfold, verbose=1)


# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('최적의 매개변수 : ', model.best_estimator_)

print('best_score_ : ', model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2스코어 : ', r2_score(y_test, y_predict))


print('걸린 시간 : ', end_time)

'''
최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=3)
best_score_ :  0.38815115592139526
model.score :  0.4868885357634498
r2스코어 :  0.4868885357634498

최적의 매개변수 :  RandomForestRegressor(max_depth=6, min_samples_leaf=9, min_samples_split=4, n_jobs=-1)
best_score_ :  0.3848639045750173
model.score :  0.5219654248215415
r2스코어 :  0.5219654248215413
걸린 시간 :  27.894556045532227
'''