from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parmeters = [
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

# 2. 모델구성

model = GridSearchCV(RandomForestClassifier(), parmeters, cv=kfold)


# 3. 컴파일, 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)

print('best_score_ : ', model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))


'''
최적의 매개변수 :  RandomForestClassifier()
best_score_ :  0.9428571428571428
model.score :  0.9111111111111111
정답률 :  0.9111111111111111
'''