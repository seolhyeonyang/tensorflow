import numpy as np
import time


# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
# model.add(Dense(100))
model.add(Dense(1))


# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

#optimizer = Adam(lr = 0.0001)
#! Adam(lr = 0.001) 이 디폴트 값
# optimizer = Adam(lr = 0.01)
# loss :  2.11741729835499e-13 결과물 :  [[11.]]

# optimizer = Adam(lr = 0.001)
# loss :  4.590274329530075e-07 결과물 :  [[10.998743]]

# optimizer = Adam(lr = 0.0001)
# loss :  4.3184780906813103e-07 결과물 :  [[10.998584]]

#optimizer = Adagrad(lr = 0.0001)
#! Adagrad(lr = 0.001) 이 디폴트 값
# optimizer = Adagrad(lr = 0.01)
# 시간 :  2.743988037109375
# loss :  1.4434823242481798e-05 결과물 :  [[11.003961]]

# optimizer = Adagrad(lr = 0.001)
# 시간 :  2.797267436981201
# loss :  1.4077581909077708e-06 결과물 :  [[10.999367]]

# optimizer = Adagrad(lr = 0.0001)
# 시간 :  2.925708293914795
# loss :  0.00034932707785628736 결과물 :  [[10.9776745]]

#optimizer = Adamax(lr = 0.0001)
#! Adamax(lr = 0.001) 이 디폴트 값
# optimizer = Adamax(lr = 0.01)
# 시간 :  3.04642653465271
# loss :  0.00047259163693524897 결과물 :  [[11.015773]]

# optimizer = Adamax(lr = 0.001)
# 시간 :  2.9192469120025635
# loss :  2.1802679839311168e-06 결과물 :  [[10.998078]]

# optimizer = Adamax(lr = 0.0001)
# 시간 :  3.0377044677734375
# loss :  0.00011430252197897062 결과물 :  [[10.984773]]

# optimizer = Adadelta(lr = 0.0001)
#! Adadelta(lr = 0.001) 이 디폴트 값
# optimizer = Adadelta(lr = 0.01)
# 시간 :  2.8618416786193848
# loss :  1.760616578394547e-05 결과물 :  [[10.999476]]

# optimizer = Adadelta(lr = 0.001)
# 시간 :  2.940138101577759
# loss :  0.00025972595904022455 결과물 :  [[10.997392]]

# optimizer = Adadelta(lr = 0.0001)
# 시간 :  2.9879746437072754
# loss :  26.420215606689453 결과물 :  [[1.8818061]]

# optimizer = RMSprop(lr = 0.0001)
#! RMSprop(lr = 0.001) 이 디폴트 값
# optimizer = RMSprop(lr = 0.01)
# 시간 :  4.878556966781616
# loss :  47.43418502807617 결과물 :  [[18.95923]]

# optimizer = RMSprop(lr = 0.001)
# 시간 :  5.028703451156616
# loss :  2.2338454723358154 결과물 :  [[13.672538]]

# optimizer = RMSprop(lr = 0.0001)
# 시간 :  4.940016746520996
# loss :  0.21562090516090393 결과물 :  [[10.190599]]

# optimizer = SGD(lr = 0.01)
#! SGD(lr = 0.01) 이 디폴트 값
# 시간 :  2.6317923069000244
# loss :  nan 결과물 :  [[nan]

# optimizer = Nadam(lr = 0.0001)
#! Nadam(lr = 0.001) 이 디폴트 값
# optimizer = Nadam(lr = 0.01)
# 시간 :  7.308628797531128
# loss :  6.591496457986068e-06 결과물 :  [[11.001377]]

# optimizer = Nadam(lr = 0.001)
# 시간 :  7.431081056594849
# loss :  3.731273068297014e-07 결과물 :  [[10.998691]]

# optimizer = Nadam(lr = 0.0001)
# 시간 :  7.327500343322754
# loss :  7.594138878630474e-05 결과물 :  [[10.98258]]

#* https://keras.io/api/optimizers/ 참고 사이트

#model.compile(loss='mse', optimizer='ada,', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

start_time = time.time()
model.fit(x, y, epochs=100, batch_size=1)
end_time = time.time()  - start_time

# 4. 평가 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('시간 : ', end_time)

print('loss : ', loss, '결과물 : ', y_pred)
# loss :  1.5347723363468707e-13 결과물 :  [[10.999998]]

