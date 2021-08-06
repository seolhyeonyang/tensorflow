from tensorflow.keras.preprocessing.text import Tokenizer


text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
#! 사용횟수가 많은것 부터 index 주고 같은 횟수면 처음부터 순서대로 한다.

x = token.texts_to_sequences([text])
print(x)        # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]
#! 수치화 된걸로 나옴

from tensorflow.keras.utils import to_categorical

word_size = len(token.word_index)
print(word_size)            #7

x = to_categorical(x)
print(x)
print(x.shape)      # (1, 9, 8)
#! 문장 1개, 단어 9개, index된거 7에 to_categorical이라 0포함