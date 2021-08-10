# 실습
#TODO 회귀 데이터를 Classifier로 만들었을 경우에 에러 확인

from sklearn.svm import LinearSVC, SVC          # 사용 가능하진 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

from sklearn.metrics import r2_score



# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# ValueError: Unknown label type: 'continuous'
#! 분류 모델 에러 발생

# model = KNeighborsRegressor()
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

# 3. 컴파일, 훈련

start_time = time.time()

model.fit(x_train, y_train)

end_time = time.time() - start_time



# 4. 평가, 예측
results = model.score(x_test, y_test)
print('걸린 시간 : ', end_time)
print('model.score : ', results)


y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
#? train_size=0.8, random_state=78
분류 모델 에러 발생

걸린 시간 :  0.0
model.score :  0.7054961415199352
r2스코어 :  0.7054961415199352

걸린 시간 :  0.009640693664550781
model.score :  0.7452210925673008
r2스코어 :  0.7452210925673008

걸린 시간 :  0.0019686222076416016
model.score :  0.7787680660887613
r2스코어 :  0.7787680660887613

걸린 시간 :  0.19137787818908691
model.score :  0.9153537212680198
r2스코어 :  0.9153537212680198
'''