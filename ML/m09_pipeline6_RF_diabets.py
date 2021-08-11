from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)


# 2. 모델 구성

from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), RandomForestRegressor())


# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2스코어 : ', r2_score(y_test, y_predict))

print('걸린 시간 : ', end_time)

'''
model.score :  0.4885777230569901
r2스코어 :  0.4885777230569901
걸린 시간 :  0.14960122108459473
'''