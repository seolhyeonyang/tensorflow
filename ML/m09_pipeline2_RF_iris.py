from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import time


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=88)


# 2. 모델구성

from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), RandomForestClassifier())



# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

print('걸린 시간 : ', end_time)

'''
model.score :  0.9333333333333333
정답률 :  0.9333333333333333
걸린 시간 :  0.08074450492858887
'''