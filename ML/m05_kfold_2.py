from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import warnings
import numpy as np
warnings.filterwarnings('ignore')


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#^ 데이터를 n_splits조각 으로 나눈다.
#! 속도가 n_splits의 배만큼 더 늘어난다.(지금은 5배)
#! 성능이 하지 않았을때와 비슷하다면 시간만 버린것이다.

# 2. 모델구성
model = LinearSVC()
# Acc : [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667         데이터 전부 kfold
# Acc : [0.95238095 1.         1.         1.         0.9047619 ] 0.9714         train 데이터만 kfold

# model = SVC()
# Acc : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667
# Acc : [0.95238095 1.         1.         0.95238095 0.95238095] 0.9714

# model = KNeighborsClassifier()
# Acc : [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
# Acc : [0.9047619  1.         1.         0.95238095 0.95238095] 0.9619

# model = LogisticRegression()
# Acc : [1.         0.96666667 1.         0.9        0.96666667] 0.9667
# Acc : [0.9047619 1.        1.        0.9047619 0.9047619] 0.9429

# model = DecisionTreeClassifier()
# Acc : [0.96666667 0.96666667 1.         0.9        0.93333333] 0.9533
# Acc : [0.95238095 1.         0.95238095 0.9047619  0.9047619 ] 0.9429

# model = RandomForestClassifier()
# Acc : [0.93333333 0.96666667 1.         0.9        0.96666667] 0.9533
# Acc : [0.9047619 1.        1.        0.9047619 0.9047619] 0.9429


# 3. 컴파일, 훈련
# 4. 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
#^ fit 과 score를 한번에 한다.

scores = cross_val_score(model, x_train, y_train, cv=kfold)
#! test data는 가만히 나두고 train data만 나눠서 교차 검증한다.

print('Acc :', scores, round(np.mean(scores),4))


