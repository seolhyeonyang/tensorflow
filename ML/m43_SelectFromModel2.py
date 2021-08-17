from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')


# 실습
#TODO 1. 그리드서치 또는 랜덤서치로 튜닝한 모델 구성 최적의 R2값과 피처임포턴스 구할것

#TODO 2. 위 스레드 값으로 SelectFromModel 돌려서 최적의 피처 갯수 구할것

#TODO 3. 위 피쳐 갯수로 피처 갯수를 조정한뒤 그걸로 다시 램덤서치, 그리드 서치해서 최적의 R2 구하기

#TODO 3. 0.47이상 만들기

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
parmeters = [
    {'n_estimators' : [100,200, 300], 'learning_rate' : [0.1, 0.001, 0.5],
    'max_depth' : [4, 5, 6], 'colsample_bytree' : [0.6, 0.9, 1], 'colsample_bylevel' : [0.6, 0.7, 0.9]},
    {'n_estimators' : [100,200, 300], 'learning_rate' : [0.01, 0.001, 0.5], 'max_depth' : [4, 5, 6]},
    {'n_estimators' : [200, 300], 'colsample_bytree' : [0.3, 0.5, 1], 'colsample_bylevel' : [0.5, 0.8, 0.9]},

]

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = GridSearchCV(XGBRegressor(), parmeters, cv=kfold, verbose=1)

# model = RandomizedSearchCV(XGBRegressor(), parmeters, cv=kfold, verbose=1)

model = XGBRegressor(colsample_bylevel= 0.9, colsample_bytree= 0.6, learning_rate= 0.1, max_depth= 6, n_estimators= 100)


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score= model.score(x_test, y_test)      # 0.2380
print('model.score : ', score)

# print('최적의 매개변수 : ', model.best_estimator_)

# print('best_params_ : ', model.best_params_)

# print('best_score_ : ', model.best_score_)

'''
XGB 모델
0.2380

램덤서치
Fitting 5 folds for each of 10 candidates, totalling 50 fits
model.score :  0.3287490500823891
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.01, max_delta_step=0, max_depth=4,
            min_child_weight=1, missing=nan, monotone_constraints='()',
            n_estimators=300, n_jobs=8, num_parallel_tree=1, random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=None)
best_params_ :  {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.01}
best_score_ :  0.46199593644350506

그리드서치
Fitting 5 folds for each of 288 candidates, totalling 1440 fits
model.score :  0.34240960249878205
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
            colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.1, max_delta_step=0, max_depth=6,
            min_child_weight=1, missing=nan, monotone_constraints='()',
            n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=None)
best_params_ :  {'colsample_bylevel': 0.9, 'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
best_score_ :  0.473156587223329
'''

threshold = np.sort(model.feature_importances_)
print(threshold)
# [0.03721908 0.04037419 0.0536947  0.05523375 0.05796364 0.06515753
#  0.10492289 0.10971878 0.21739212 0.25832343]

for thresh in threshold:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(colsample_bylevel= 0.9, colsample_bytree= 0.6, learning_rate= 0.1, max_depth= 6, n_estimators= 100)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))


