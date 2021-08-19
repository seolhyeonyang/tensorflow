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

#########################################################
#* 라벨 통합
#########################################################
print('='*100)

# for index, value in enumerate(y):
#     if value == 9:
#         y[index] = 7
#     elif  value == 3:
#         y[index] = 5
#     elif value == 8:
#         y[index] = 7
#     elif  value == 4:
#         y[index] = 5

for index, value in enumerate(y):
    if value == 3 :
        y[index] = 4
    elif  value == 5 :
        y[index] = 6
    elif value == 7 :
        y[index] = 8
    elif value == 9 :
        y[index] = 8



print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle=True, random_state=78, stratify=y)

# print(pd.Series(y_train).value_counts())

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

smote = SMOTE(random_state=78)

start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
end_time = time.time() - start_time


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

print('smote 시간 : ', end_time)


'''
smote 전 :  (3673, 11) (3673,)
smote 후 :  (9888, 11) (9888,)
smote전 레이블 값 분포 :
6.0    1648
5.0    1093
7.0     660
8.0     135
4.0     122
3.0      15
dtype: int64
smote후 레이블 값 분포 :
6.0    1648
5.0    1648
4.0    1648
8.0    1648
7.0    1648
3.0    1648
dtype: int64
smote 전 model.score :  0.6587755102040816
smote 후 model2.score :  0.6236734693877551
'''

