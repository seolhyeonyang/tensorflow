from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.metrics import accuracy_score


datasets = load_breast_cancer()

# print(datasets.DESCR)       # 데이터 내용 (DESCR-묘사하다.)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)     # (569, 30) (569,)

# print(y[:20])       # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# print(np.unique(y))     #[0 1]  y에 어떤 값이 있는지


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#print(x_train.shape, x_test.shape)      #(455, 30) (114, 30)


# 2. 모델구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()


# 3. 컴파일, 훈련

start_time = time.time()

model.fit(x_train, y_train)

end_time = time.time() - start_time



# 4. 평가, 예측
results = model.score(x_test, y_test)
print('걸린 시간 : ', end_time)
print('model.score : ', results)


#print('+'*10,' 예측 ', '+'*10)
y_predict = model.predict(x_test[:50])
# print(y_predict)
# print(y_test[:5])

acc = accuracy_score(y_test[:50], y_predict)
print('accuracy_score : ', acc)


""" 
#? train_size=0.8, random_state=71, MinMaxScaler()
걸린 시간 :  0.0
model.score :  0.9824561403508771
accuracy_score :  0.98

걸린 시간 :  0.006980180740356445
model.score :  0.9736842105263158
accuracy_score :  0.98

걸린 시간 :  0.0
model.score :  0.9736842105263158
accuracy_score :  0.98

걸린 시간 :  0.018064022064208984
model.score :  0.9736842105263158
accuracy_score :  0.98

걸린 시간 :  0.004018545150756836
model.score :  0.9649122807017544
accuracy_score :  0.98

걸린 시간 :  0.12569379806518555
model.score :  0.9736842105263158
accuracy_score :  0.98
"""
