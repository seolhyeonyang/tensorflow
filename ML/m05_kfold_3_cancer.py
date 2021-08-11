from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target


# 1. 데이터
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델구성
# model = LinearSVC()
# Acc : [0.78947368 0.93859649 0.9122807  0.92982456 0.97345133] 0.9087

# model = SVC()
# Acc : [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 0.921

# model = KNeighborsClassifier()
# Acc : [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928

# model = LogisticRegression()
# Acc : [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 0.9385

# model = DecisionTreeClassifier()
# Acc : [0.92982456 0.94736842 0.92982456 0.88596491 0.92920354] 0.9244

model = RandomForestClassifier()
# Acc : [0.97368421 0.95614035 0.95614035 0.96491228 0.98230088] 0.9666


# 3. 컴파일, 훈련
# 4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)

print('Acc :', scores, round(np.mean(scores),4))







""" 

"""
