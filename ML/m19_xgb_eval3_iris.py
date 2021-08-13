from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


# 1. 데이터
datasets = load_iris()

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
model = XGBClassifier(n_estimators=100, learning_rate=0.05, n_jobs=-1)


# 3. 훈련
model.fit(x_train, y_train, verbose=1,  eval_metric=['mlogloss'], eval_set=[(x_train, y_train), (x_test, y_test)])
#^ Classifier는  eval_metric='mlogloss'가 디폴트



# 4. 평가
results = model.score(x_test, y_test)
print('results : ', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

print('=====================================')
hist = model.evals_result()
print(hist)



# 시각화

plt.plot(hist['validation_0']['mlogloss'])
plt.plot(hist['validation_1']['mlogloss'])

plt.title('rmse')
plt.xlabel('n_estimators')
plt.ylabel('rmse, val_rmse')
plt.legend('train, val')

plt.show()