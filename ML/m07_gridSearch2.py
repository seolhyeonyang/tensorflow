# 실습
#TODO m07_1 최적의 파라미터값을 가지고 model를 구성

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


# 2. 모델구성

model = SVC(C=1, kernel='linear')


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))


y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

# model.score :  0.9777777777777777
# 정답률 :  0.9777777777777777