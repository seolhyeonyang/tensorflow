import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate  # 소문자는 메소드, 대문자는 클래스 (네이밍 룰이다.)
from sklearn.model_selection import train_test_split


x1 = np.array([range(100), range(301,401), range(1,101)])
# x2 = np.array([range(101,201), range(411,511), range(100,200)])

x1 = np.transpose(x1)
# x2 = np.transpose(x2)

# y = np.array([range(1001,1101)])
# y = np.transpose(y)
y1 = np.array(range(1001,1101))
y2 = np.array(range(1901,2001))

# print(x1.shape, x2.shape, y1.shape, y2.shape)     # (100, 3) (100, 3) (100,) (100,)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.7, shuffle=True, random_state=66)

# print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape, y2_train.shape, y2_test.shape)    
# (70, 3) (30, 3) (70,) (30,) (70,) (30,)


# 2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3) 

'''
# 2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)

merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)

# last_output = Dense(1)(merge3)
''' 
output21 = Dense(7)(output1) 
last_output1 = Dense(1, name='last_output1')(output21)

output22 = Dense(8)(output1)
last_output2 = Dense(1, name='last_output2')(output22)


model = Model(inputs = input1, outputs=[last_output1, last_output2])
#  두가지 모델이 하나로 합쳐졌다가 다시 나눠져 2가지 결과로 나온다.

model.summary()


#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

#4 평가, 예측 -> 두개를 선택적으로 해도 된다.
results = model.evaluate(x1_test, [y1_test, y2_test])
# print(results)
print("loss : ", results[0])
print("metrics['mae'] : ", results[1])

