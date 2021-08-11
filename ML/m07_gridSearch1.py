from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
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
    {'C':[1, 10, 100, 1000], 'kernel':['linear']},
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}
]
#! 여러개의 파라미터를 정의 하려면 list[]로 해준다.

# 2. 모델구성

model = GridSearchCV(SVC(), parmeters, cv=kfold)
#! 기존 모델에 내가 쓸 수 있는 파라미터를 딕셔너리 구조로 정의해 랩핑해준다.


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)
#! model.best_estimator_ 모델에서 가장 좋은 평가를 알려준다.
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
#! 경우의 수 90번 중에 위에가 가장 좋다.

print('best_score_ : ', model.best_score_)
#! model.best_score_ 모델에서 가장 좋은 score를 알려준다.
# best_score_ :  0.9800000000000001
#^ cross_val을 한 데이터의 값이다.(fit한 데이터들이다.) train값의 최상의 값
#! best는 GridSearchCV에서만 사용 가능

print('model.score : ', model.score(x_test, y_test))
# model.score :  0.9777777777777777

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# best_score_ :  0.980952380952381
# 정답률 :  0.9777777777777777
#^ 평가용으로 fit을 하지 않은 데이터의 값       test값의 최상의 값