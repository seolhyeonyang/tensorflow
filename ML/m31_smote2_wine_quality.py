from imblearn.over_sampling import SMOTE
from pandas.core.algorithms import value_counts
#! 데이터 증폭 (제일 많은 데이터에 맞춰 증폭해 준다.) - F1 스코어에서 효과를 얻을 수 있다.
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

datasets = pd.read_csv('/study2/_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]

print(x.shape, y.shape)     # (4898, 11) (4898,)

# print(x_new.shape, y_new.shape)     # (148, 13) (148,)
# print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, 
                                                    shuffle=True, random_state=66, stratify=y)

# print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier()

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('model.score : ', score)

'''
XGB 디폴트 모델
model.score :  0.643265306122449
'''

########################################### smote 적용 ##########################################

print('################### smote 적용 ###################')

smote = SMOTE(random_state=66, k_neighbors=3)
#! 라벨 데이터 개수가 k_neighbors 보다 작으면 증폭이 되지 않는다. (라벨 9의 데이터개수는 4개)
#^ k_neighbors 디폴트는 5
#! k_neighbors를 최소 데이터수 보다 작게 준다.

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())
# 0    53
# 1    53
# 2    53

# print(x_smote_train.shape, y_smote_train.shape)     # (111, 13) (111,)  -> (159, 13) (159,)

model2 = XGBClassifier()

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print('model2.score : ', score2)


print('smote 전 : ', x_train.shape, y_train.shape)
print('smote 후 : ', x_smote_train.shape, y_smote_train.shape)
print('smote전 레이블 값 분포 :\n', pd.Series(y_train).value_counts())
print('smote후 레이블 값 분포 :\n', pd.Series(y_smote_train).value_counts())

print('smote 전 model.score : ', score)
print('smote 후 model2.score : ', score2)



