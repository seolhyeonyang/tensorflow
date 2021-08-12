from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 데이터
datasets = load_diabetes()

diabetes_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

# print(diabetes_df)
# print(diabetes_df.columns)

diabetes_df = diabetes_df.drop(['age', 'sex', 's1','s4'], axis=1)

x = diabetes_df.values
# print(diabetes_df.columns)

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
r2  :  -0.3247143175883884
[0.06666561 0.0055428  0.22729414 0.1201307  0.03887656 0.05632917
 0.05115188 0.00264899 0.35660042 0.07475972]

*변경후 (삭제)
r2  :  -0.28445730116611156
[0.22583846 0.1217319  0.08772868 0.07566674 0.40350777 0.08552644]

#? 2. RandomForestClassifier
*기존
r2  :  0.37822914385243744
[0.06537841 0.01150993 0.28092297 0.10511991 0.04575597 0.05476815
 0.04872106 0.01874404 0.2918542  0.07722536]

*변경후
r2  :  0.3666717051911186
[0.29578155 0.12312702 0.09861307 0.06853094 0.32721039 0.08673704]

#? 3. GradientBoostingClassifier
*기존
r2  :  0.3863897672312707
[0.06025019 0.01196654 0.27516557 0.11701368 0.02497993 0.05276835
 0.04104457 0.01604811 0.34318757 0.05757549]

*변경후
r2  :  0.3566487038984385
[0.30232495 0.11902725 0.09624623 0.04800484 0.36625836 0.06813836]

#? 4. XGBClassifier
*기존
r2  :  0.23802704693460175
[0.02593722 0.03821947 0.19681752 0.06321313 0.04788675 0.05547737
 0.07382318 0.03284872 0.3997987  0.06597802]

*변경후
r2  :  0.3008522300150823
[0.21938416 0.09112234 0.09656683 0.0946343  0.4103561  0.08793634]
'''