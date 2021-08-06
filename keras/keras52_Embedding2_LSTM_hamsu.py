from numpy.lib.arraypad import pad
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.layers.recurrent import LSTM


docs = ['너무 재밋어요', '참 최고예요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고 싶네요', '글세요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요']


# 긍정 1, 부정 2
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
#! 크기가 다른 데이터를 크기 맞춰 주기 위해 쓴다.
#! 작은 데이터에 0을 넣어 크기 맞춰준다.

pad_x = pad_sequences(x, padding='pre', maxlen=5)   # padding = 'post' -> 뒤에 채워주는것

print(pad_x)
print(pad_x.shape)      # (13, 5)

word_size = len(token.word_index)
print(word_size)        # 27

print(np.unique(pad_x))

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

# 1. 모델 구성

# 순차형
# model = Sequential()
# model.add(Embedding(input_dim = 28, output_dim=11, input_length=5))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,973
Trainable params: 5,973
Non-trainable params: 0
_________________________________________________________________
'''

# 함수형
input = Input(shape=(5,))
embedding = Embedding(input_dim = 28, output_dim=11)(input)
lstm = LSTM(32)(embedding)
dense = Dense(1)(lstm)

model = Model(inputs=input, outputs=dense)
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,973
Trainable params: 5,973
Non-trainable params: 0
_________________________________________________________________
'''

model.summary()



""" # 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=100, batch_size=10)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ',  acc) """
