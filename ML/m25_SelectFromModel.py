from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel


# 1. 데이터
# datasets = load_boston()
# x = datasets.data
# y = datasets.target

x, y = load_boston(return_X_y=True)
#! 디폴트는 False 데이터를 통으로 반환

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
model = XGBRegressor(n_jobs=8)


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score= model.score(x_test, y_test)      # 0.9221
print('model.score : ', score)

threshold = np.sort(model.feature_importances_)
print(threshold)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

for thresh in threshold:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #! SelectFromModel도 모델이다. (컬럼을 자동 삭제 해주는 모델)
    #^ model를 돌리는데 thresh는 빼고 돌린다.(여기에서는 model = XGBRegressor, 
    #^ threshold = 0.00134153 이상인것으로 모델을 제구성(0.001 이상인 컬럼만 사용)), 들어가는 숫자 이상인 컬럼만으로 재구성
    # print(selection)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))


'''
#! 과적합 처리(중요!!, 필수로 해야함)
1. 훈련 데이터 증가
2. Drop out (딥러닝에서)
3. 정규화
    (regularization,
    normalization)
4. 피쳐(컬럼)을 줄인다.
    (피쳐가 필요 없다고 확신이 생겼을때)

'''