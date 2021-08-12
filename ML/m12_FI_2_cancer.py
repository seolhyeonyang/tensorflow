from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. 데이터
datasets = load_breast_cancer()

cancer_df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

# print(cancer_df)
# print(cancer_df.columns)

cancer_df = cancer_df.drop(['compactness error', 'smoothness error', 'mean fractal dimension',
                            'texture error', 'concavity error', 'symmetry error'], axis=1)

x = cancer_df.values

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
acc  :  0.9210526315789473
[0.         0.05940707 0.         0.         0.         0.
 0.         0.04216086 0.         0.         0.01702306 0.01405362
 0.         0.         0.00624605 0.         0.         0.00433754
 0.         0.         0.         0.01612033 0.         0.71474329
 0.         0.         0.00461856 0.11660508 0.00468454 0.        ]

*변경후 
acc  :  0.9298245614035088
[0.00468454 0.05940707 0.         0.         0.         0.
 0.02248579 0.01967507 0.         0.         0.         0.01233852
 0.00433754 0.01405362 0.         0.01612033 0.         0.71942783
 0.         0.         0.00461856 0.11582433 0.         0.00702681]

#? 2. RandomForestClassifier
*기존
acc  :  0.956140350877193
[0.01225062 0.01859238 0.03958346 0.03669728 0.00478283 0.00694123
 0.0476587  0.07275833 0.00489757 0.00343528 0.03307357 0.00417455
 0.0146074  0.04986522 0.00395151 0.00384977 0.00438939 0.00740723
 0.00444572 0.0049896  0.12127495 0.0197467  0.1618774  0.14725941
 0.01282442 0.01521444 0.03330252 0.09486201 0.00838059 0.00690594]

*변경후
acc  :  0.9649122807017544
[0.02817877 0.02000987 0.03478098 0.03829235 0.00530283 0.01355789
 0.03680762 0.11881133 0.00520481 0.01823991 0.00637891 0.03404885
 0.00925685 0.00574273 0.11863818 0.01854403 0.13589131 0.15558345
 0.01713693 0.00809829 0.02643066 0.12977009 0.007406   0.00788737]

#? 3. GradientBoostingClassifier
*기존
acc  :  0.9473684210526315
[6.08660682e-04 3.69721461e-02 3.89063689e-04 2.05254577e-03
 9.50341654e-04 2.13707872e-06 7.60165433e-04 1.28005958e-01
 1.49854769e-03 4.51105397e-03 4.00642930e-03 6.36118273e-05
 8.66114889e-04 1.78717224e-02 1.42385691e-03 2.53428814e-03
 1.00987972e-02 7.23690072e-04 3.00774637e-05 8.83913313e-04
 3.14129941e-01 4.20992132e-02 3.61779341e-02 2.71582071e-01
 5.52869272e-03 1.30441729e-04 1.35917706e-02 1.00662728e-01
 2.98382348e-05 1.81424823e-03]

*변경후
acc  :  0.9473684210526315
[3.74531455e-04 3.53722982e-02 5.77148780e-04 2.65167973e-03
 2.19194879e-03 1.92480084e-04 4.81029135e-03 1.24208168e-01
 1.66369106e-03 3.92700436e-03 1.49214798e-04 1.82155615e-02
 9.79499140e-04 9.43706107e-04 3.32038699e-01 4.35665871e-02
 4.44749033e-02 2.61018056e-01 4.65478400e-03 6.16793455e-05
 1.39942452e-02 1.02865075e-01 2.40702833e-05 1.04467756e-03]

#? 4. XGBClassifier
*기존
acc  :  0.9736842105263158
[0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
 0.0054994  0.09745205 0.00340272 0.00369179 0.00769184 0.00281184
 0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
 0.00639412 0.0050556  0.01813928 0.02285903 0.22248562 0.28493083
 0.00233393 0.         0.00903706 0.11586285 0.00278498 0.00775311]

*변경후
acc  :  0.9824561403508771
[0.01120542 0.03520589 0.         0.02403574 0.00438002 0.01350832
 0.00457921 0.13570914 0.00133568 0.00826551 0.01102016 0.01234619
 0.00384806 0.00448597 0.0108443  0.01874688 0.2885436  0.2846242
 0.00183119 0.00093789 0.01037866 0.09786308 0.0040358  0.01226908]
'''
