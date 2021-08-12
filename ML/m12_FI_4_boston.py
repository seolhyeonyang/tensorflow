from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 데이터
datasets = load_boston()

boston_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

# print(boston_df)
# print(boston_df.columns)

boston_df = boston_df.drop(['CHAS', 'ZN', 'AGE', 'B'], axis=1)

x = boston_df.values
print(boston_df.columns)

x_train, x_test, y_train, y_test = train_test_split(x, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
r2 = model.score(x_test, y_test)
print('r2  : ', r2)

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
r2  :  0.8034240877221178
[0.04160495 0.00229187 0.00951587 0.         0.01655626 0.27984775
 0.01157659 0.0609203  0.00380709 0.02525359 0.0077901  0.00700583
 0.53382981]

*변경후 (삭제)
r2  :  0.8132124163094973
[0.06248107 0.01132072 0.01898531 0.28135594 0.0637753  0.00347996
 0.00933965 0.00959066 0.53967138]

#? 2. RandomForestClassifier
*기존
r2  :  0.923704725971418
[0.03951796 0.00152059 0.00686142 0.00084132 0.02109882 0.35844115
 0.01430013 0.06708264 0.00490086 0.01604788 0.01864994 0.01188683
 0.43885046]

*변경후
r2  :  0.9159385281304138
[0.04952963 0.0079544  0.02259724 0.39951287 0.06595663 0.00640958
 0.01519931 0.01953799 0.41330235]

#? 3. GradientBoostingClassifier
*기존
r2  :  0.9458156865685029
[2.38829323e-02 1.59073367e-04 2.00770439e-03 2.00989506e-04
 4.09208792e-02 3.57675277e-01 6.73406320e-03 8.43662728e-02
 2.32453147e-03 1.13949176e-02 3.37008138e-02 6.49322527e-03
 4.30139320e-01]

*변경후
r2  :  0.9392282416865843
[0.02783099 0.00350103 0.04474976 0.36296264 0.08154814 0.00305337
 0.01239604 0.03437536 0.42958266]

#? 4. XGBClassifier
*기존
r2  :  0.9221188601856797
[0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643
 0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
 0.42848358]

*변경후
r2  :  0.9307724288278274
[0.01878339 0.02460936 0.06401006 0.30338958 0.06172328 0.0111962
 0.03891074 0.04787611 0.42950124]
'''