from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
import time


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

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
    {'n_jobs' : [-1], 'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [3,5, 7, 10]},
    {'n_jobs' : [-1], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [3, 6, 9, 11], 'min_samples_split' : [3, 4, 5]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7], 'min_samples_split' : [3, 4, 5]},
    {'n_jobs' : [-1], 'min_samples_split' : [2, 3, 5, 10]}
]

# 2. 모델 구성

# model = GridSearchCV(RandomForestRegressor(), parmeters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 73 candidates, totalling 365 fits

model = RandomizedSearchCV(RandomForestRegressor(), parmeters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('최적의 매개변수 : ', model.best_estimator_)

print('best_params_ : ', model.best_params_)

print('best_score_ : ', model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2스코어 : ', r2_score(y_test, y_predict))

print('걸린 시간 : ', end_time)

'''
GridSearchCV
최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=-1)
best_params_ :  {'min_samples_split': 5, 'n_jobs': -1}
best_score_ :  0.8344621185551514
model.score :  0.918106770188637
r2스코어 :  0.9181067701886368
걸린 시간 :  32.23291015625

RandomizedSearchCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 :  RandomForestRegressor(max_depth=6, min_samples_leaf=3, n_estimators=200, n_jobs=-1)
best_params_ :  {'n_jobs': -1, 'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 6}
best_score_ :  0.8030514257827409
model.score :  0.9220061607067438
r2스코어 :  0.9220061607067439
걸린 시간 :  5.674943208694458
'''