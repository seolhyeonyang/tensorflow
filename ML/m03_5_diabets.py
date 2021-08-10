from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

from sklearn.svm import LinearSVC, SVC          # 사용 가능하진 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442,10), (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
#! 분류 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#! 회귀 모델
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

# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score : ', acc)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
#? train_size=0.8, random_state=9
걸린 시간 :  0.06981348991394043
model.score :  0.02247191011235955
accuracy_score :  0.02247191011235955
r2스코어 :  0.2548227176977842

걸린 시간 :  0.07579684257507324
model.score :  0.011235955056179775
accuracy_score :  0.011235955056179775
r2스코어 :  -0.4100755437506187

걸린 시간 :  0.0
model.score :  0.0
accuracy_score :  0.0
r2스코어 :  -0.11672945237496823

걸린 시간 :  0.10368967056274414
model.score :  0.0
accuracy_score :  0.0
r2스코어 :  0.04247415700715729

걸린 시간 :  0.008975982666015625
model.score :  0.011235955056179775
accuracy_score :  0.011235955056179775
r2스코어 :  0.029210630697685258

걸린 시간 :  0.3071732521057129
model.score :  0.0
accuracy_score :  0.0
r2스코어 :  0.08885314073842632

#^ 회귀모델
걸린 시간 :  0.0
model.score :  0.4904172186988308
r2스코어 :  0.4904172186988308

걸린 시간 :  0.00994420051574707
model.score :  0.5851141269959738
r2스코어 :  0.5851141269959738

걸린 시간 :  0.002028942108154297
model.score :  -0.2464431884261178
r2스코어 :  -0.2464431884261178

걸린 시간 :  0.16156649589538574
model.score :  0.555799951681768
r2스코어 :  0.555799951681768
'''