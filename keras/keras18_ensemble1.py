import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate  # 소문자는 메소드, 대문자는 클래스 (네이밍 룰이다.)
from sklearn.model_selection import train_test_split


x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001,1101))
# y = np.array([range(1001,1101)])
# y = np.transpose(y)

# print(x1.shape, x2.shape, y.shape)     # (100, 3) (100, 3) (100,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=66)
# test_size의 디폴트가 0.25이기 때문에 자연스레 train_size의 디폴트 값 0.75 
# print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)    # (70, 3) (30, 3) (70, 3) (30, 3) (70, 1) (30, 1)


# 2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3) 
# x1, y1 모델

# 2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)
# x2, y1 모델


# merge1 = concatenate([output1, output2])    # concatenate -> 사슬처럼 엮다. output1,2 를 엮는다. layer다.
# concatenate(data, axis = 1) 기본 구조
merge1 = Concatenate()([output1, output2])
# Concatenate(axis = 1),(data) 기본 구조
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)

last_output = Dense(1)(merge3) # 여기가 최종 output!! output1,2는 히든이다. 그래서 최종 output전에는 모두 히든이라 하이퍼 파라미터 튜닝이 가능하다. 

model = Model(inputs = [input1, input2], outputs=last_output)

# model.summary()

'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense11 (Dense)                 (None, 10)           40          input_2[0][0]
__________________________________________________________________________________________________
dense1 (Dense)                  (None, 10)           40          input_1[0][0]
__________________________________________________________________________________________________
dense12 (Dense)                 (None, 10)           110         dense11[0][0]
__________________________________________________________________________________________________
dense2 (Dense)                  (None, 7)            77          dense1[0][0]
__________________________________________________________________________________________________
dense13 (Dense)                 (None, 10)           110         dense12[0][0]
__________________________________________________________________________________________________
dense3 (Dense)                  (None, 5)            40          dense2[0][0]
__________________________________________________________________________________________________
dense14 (Dense)                 (None, 10)           110         dense13[0][0]
__________________________________________________________________________________________________
output1 (Dense)                 (None, 11)           66          dense3[0][0]
__________________________________________________________________________________________________
output2 (Dense)                 (None, 12)           132         dense14[0][0]
__________________________________________________________________________________________________     concatenate는 그냥 output1,2 더해준것이다.
concatenate (Concatenate)       (None, 23)           0           output1[0][0]
                                                                 output2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           240         concatenate[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5)            55          dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            6           dense_1[0][0]
==================================================================================================
Total params: 1,026
Trainable params: 1,026
Non-trainable params: 0
__________________________________________________________________________________________________      최종 output 전까지는 모두 hidden이다. 그래서 튜닝이 가능하다.
'''

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1)
# x1_train이 모델 1을 x2_train이 모델 2 실행
# 순서가 바뀌면 적용 모델도 바뀐다 따라서 값도 바뀐다.

#4 평가, 예측 -> 두개를 선택적으로 해도 된다.
results = model.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])
print("metrics['mae'] : ", results[1])
# evaluate는 loss와 metrics의 값을 출력해 준다.