from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 데이터
datasets = load_wine()

wine_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

# print(wine_df)
# print(wine_df.columns)

wine_df = wine_df.drop(['proanthocyanins', 'total_phenols', 'ash', 'alcalinity_of_ash', 'nonflavanoid_phenols'], axis=1)

x = wine_df.values

x_train, x_test, y_train, y_test = train_test_split(x, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier()
# model =RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc  : ', acc)

print(model.feature_importances_)

# print(datasets.feature_names)

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
acc  :  0.9166666666666666
[0.00489447 0.01598859 0.         0.         0.04078249 0.
 0.1569445  0.         0.         0.03045446 0.0555874  0.33215293
 0.36319516]

*변경후 ( 삭제)
acc  :  0.9722222222222222
[0.00489447 0.         0.04078249 0.18739896 0.01598859 0.0555874
 0.33215293 0.36319516]

#? 2. RandomForestClassifier
*기존
acc  :  1.0
[0.12944864 0.03149691 0.02168403 0.02541618 0.02772879 0.05316488
 0.15590626 0.01464686 0.03019534 0.16105305 0.10653913 0.10260825
 0.14011168]

*변경후
acc  :  1.0
[0.13230311 0.03899895 0.03459675 0.16436593 0.15239155 0.11772674
 0.14701069 0.21260629]

#? 3. GradientBoostingClassifier
*기존
acc  :  0.9722222222222222
[1.68092916e-02 4.07210320e-02 2.03983445e-02 5.03101599e-03
 5.82598005e-03 3.09684944e-05 1.04150450e-01 1.28879362e-04
 7.07674635e-05 2.49546037e-01 2.82703536e-02 2.50853534e-01
 2.78163346e-01]

*변경후
acc  :  1.0
[0.02215115 0.04595043 0.0047712  0.12290978 0.25578758 0.02630601
 0.23883536 0.28328849]

#? 4. XGBClassifier
*기존
acc  :  1.0
[0.01854127 0.04139536 0.01352911 0.01686821 0.02422602 0.00758254
 0.10707161 0.01631111 0.00051476 0.12775211 0.01918284 0.50344414
 0.10358089]

*변경후
acc  :  1.0
[0.0266574  0.0384905  0.03015656 0.15705651 0.17699434 0.03101406
 0.38758248 0.15204813]
'''
