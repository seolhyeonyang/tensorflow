from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = XGBRegressor(n_estimators=20, learning_rate=0.05, n_jobs=-1)


# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], eval_set=[(x_train, y_train), (x_test, y_test)])
#! 평가하는걸 넣어줘야 verbose가 보임 eval_set=[(훈련셋), (검증셋)], eval_metric= 평가 지표 지정 가능
#^ Regressor는  eval_metric='rmse'가 디폴트
#* eval_metric=['rmse', 'mae', 'logloss'] Regressor의 평가지표

# 4. 평가
results = model.score(x_test, y_test)
print('results : ', results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

'''
results :  0.9185556547141065
r2 :  0.9185556547141065
'''

print('=====================================')
hist = model.evals_result()
print(hist)
print(hist['validation_0']['rmse'])
print(hist['validation_1']['rmse'])



# 시각화

# plt.plot(hist['validation_0']['rmse'])
# plt.plot(hist['validation_1']['rmse'])

# plt.title('rmse')
# plt.xlabel('n_estimators')
# plt.ylabel('rmse, val_rmse')
# plt.legend('train, val')

# plt.show()



print('=========== 선생님 그래프================')

epochs = len(hist['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, hist['validation_0']['rmse'], label='Train')
ax.plot(x_axis, hist['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()

# fig, ax = plt.subplots()
# ax.plot(x_axis, hist['validation_0']['rmse'], label='Train')
# ax.plot(x_axis, hist['validation_1']['rmse'], label='Test')
# ax.legend()
# plt.ylabel('Rmse')
# plt.title('XGBoost RMSE')
# plt.show()
