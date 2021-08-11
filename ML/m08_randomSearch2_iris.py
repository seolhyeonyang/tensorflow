from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import time


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터
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


# 2. 모델구성

# model = GridSearchCV(RandomForestClassifier(), parmeters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 67 candidates, totalling 335 fits

model = RandomizedSearchCV(RandomForestClassifier(), parmeters, cv=kfold, verbose=1)
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
print('정답률 : ', accuracy_score(y_test, y_predict))

print('걸린 시간 : ', end_time)

'''
GridSearchCV
최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=10, n_estimators=200, n_jobs=-1)
best_params_ :  {'max_depth': 6, 'min_samples_leaf': 10, 'n_estimators': 200, 'n_jobs': -1}
best_score_ :  0.9523809523809523
model.score :  0.9555555555555556
정답률 :  0.9555555555555556
걸린 시간 :  29.406254529953003

RandomizedSearchCV
최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
best_params_ :  {'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_depth': 10}
best_score_ :  0.9523809523809523
model.score :  0.9333333333333333
정답률 :  0.9333333333333333
걸린 시간 :  5.736478090286255
'''