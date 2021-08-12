from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#! DecisionTreeClassifier의 확장형이 RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# 1. 데이터
datasets = load_iris()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier(max_depth=5)
model = RandomForestClassifier()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc  : ', acc)

print(model.feature_importances_)
#! feature_importances_은 Tree계열에서 제공해 준다.
# [0.0125026  0.         0.53835801 0.44913938] -> train_size=0.8
#! 컬럼별 acc 기여도(중요도) -> 모두 합치면 1이다. 0인것은 빼도 된다.
#^ 절대적인 것은 아니다. 데이터를 바꿀때 마다 변환다.
# [0.         0.01906837 0.41726222 0.56366941] -> train_size=0.7

def plot_feature_importances_dataset(model):
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_,
            align = 'center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()