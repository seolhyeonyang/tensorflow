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
print(word_size)        # 27            0부터니깐 입력하려면 28이다.

print(np.unique(pad_x))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# 1. 모델 구성
model = Sequential()

model.add(Embedding(input_dim = 28, output_dim=11, input_length=5))
# model.add(Embedding(28, 11))
# model.add(Embedding(28, 11, input_length=5))
# 원핫인코딩을 하면 shape 가    (13, 5) -> (13, 5, 27)
# 데이터가 너무 커진다.
#! 그래서 데이터를 벡터화 시킨다. (Embedding) - 원핫인코딩 할 필요없다.
#^ input_dim = 라벨 개수(단어사전의 개수), output_dim = 아웃풋 노드 개수, input_lenght = 단어수, 길이
#* input_dim은 데이터 라벨보다 작게하면 오류나고 크게하면 돌아가긴한다.(크게 하면 연산을 더 많이 하게 된다.)

model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
#! model.add(Embedding(input_dim = 27, output_dim=11, input_length=5))

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================   input_dim * output_dim
embedding (Embedding)        (None, 5, 11)             297
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,962
Trainable params: 5,962
Non-trainable params: 0
_________________________________________________________________
'''

'''
#! model.add(Embedding(27, 11))

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 11)          297
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,962
Trainable params: 5,962
Non-trainable params: 0
_________________________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(pad_x, labels, epochs=100, batch_size=10)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ',  acc)
