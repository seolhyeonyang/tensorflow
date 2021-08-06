import numpy as np
from tensorflow.python.keras.saving.model_config import model_from_config

#1. 데이터
x = np.array([range(100), range(301,401), range(1,101),
            range(100), range(401,501)])
x = np.transpose(x)

print(x.shape)      # (100, 5)

y = np.array([range(711,811), range(101,201)])
y = np.transpose(y)

print(y.shape)      # (100, 2)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model       # Model 함수형 모델에 사용
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(5,))
xx = Dense(3)(input1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)
# 레이어명 똑같이 가능 summary에서 구분하려면 name =''으로 설정

model = Model(inputs = input1, outputs=output1)

model.summary()


#3. 컴파일, 훈련
#4. 평가, 예측 
