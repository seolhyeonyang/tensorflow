from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#! 데이터를 n_splits조각 으로 나눈다.


# 2. 모델구성
# model = LinearSVC()
# Acc : [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667

# model = SVC()
# Acc : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667

# model = KNeighborsClassifier()
# Acc : [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96

# model = LogisticRegression()
# Acc : [1.         0.96666667 1.         0.9        0.96666667] 0.9667

# model = DecisionTreeClassifier()
# Acc : [0.96666667 0.96666667 1.         0.9        0.93333333] 0.9533

model = RandomForestClassifier()

# 3. 컴파일, 훈련
# 4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
#! fit 과 score를 한번에 한다.

print('Acc :', scores, round(np.mean(scores),4))


