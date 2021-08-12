# 실습
#TODO 피쳐임포턴스가 정체 중요도에서 20% 미만인 컬럼들을 제거하여 데이터셋을 재 구성후
#TODO 각 모델 별로 도려서 결과 도출

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 데이터
datasets = load_iris()
# x = datasets.data
# y = datasets.target

# print(x)
# print(y)
# print(datasets.feature_names)

iris_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

iris_df = iris_df.drop(['sepal width (cm)', 'sepal length (cm)'], axis=1)

# print(iris_df)
x = iris_df.values

x_train, x_test, y_train, y_test = train_test_split(x, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc  : ', acc)

print(model.feature_importances_)

# def plot_feature_importances_dataset(model):
#     n_feature = datasets.data.shape[1]
#     plt.barh(np.arange(n_feature), model.feature_importances_,
#             align = 'center')
#     plt.yticks(np.arange(n_feature), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_feature)

# plot_feature_importances_dataset(model)
# plt.show()


'''
결과 비교
#? 1. DecisionTreeClassifier
*기존
acc  :  0.9333333333333333
[0.0125026  0.         0.53835801 0.44913938]

*변경후 ('sepal width (cm)', 'sepal length (cm)' 삭제)
acc  :  0.9333333333333333
[0.54517411 0.45482589]

#? 2. RandomForestClassifier
*기존
acc  :  0.9333333333333333
[0.10154069 0.03043154 0.47944533 0.38858244]

*변경후
acc  :  0.9666666666666667
[0.52215661 0.47784339]

#? 3. GradientBoostingClassifier
*기존
acc  :  0.9333333333333333
[0.00360582 0.01222095 0.30468344 0.67948978

*변경후
acc  :  0.9666666666666667
[0.22466574 0.77533426]

#? 4. XGBClassifier
*기존
acc  :  0.9
[0.01835513 0.0256969  0.62045246 0.3354955 ]

*변경후
acc  :  0.9666666666666667
[0.510896   0.48910394]
'''