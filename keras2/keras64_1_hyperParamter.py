import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x= Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x= Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x= Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'batch_size' : batches,'optimizer' : optimizers,
            'drop' : dropout}

hyperparamters = create_hyperparameter()
# print(hyperparamters)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#! tf 모델을 sklearn형태로 램핑해주는것

# model2 = build_model()
#^ 위처럼 해주면 안된다.
model2 = KerasClassifier(build_fn=build_model, verbose=1)#, epochs=2, validation_split=0.2)
#^ epochs / validation_split 적용 가능

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# model = RandomizedSearchCV(model2, hyperparamters, cv=5)
model = GridSearchCV(model2, hyperparamters, cv=2)
#^ 크로스 발리데이션에 숫자만 입력해도 가능, epochs는 적용 안된다.
#! GridSearchCV, RandomizedSearchCV는 tensorflow 모델을 쓸 수 없다.
#! tf모델을 sklearn로 랩핑해준다.

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)
#^ epochs / validation_split 적용 가능
#! KerasClassifier와 같이 epochs 주면 fit이 우선순위다

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
'''
{'batch_size': 1000, 'drop': 0.1, 'optimizer': 'adam'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002C2D7AF6AF0>
0.9458333253860474
'''
acc = model.score(x_test, y_test)
print('최종 스코어 : ', acc)
# 최종 스코어 :  0.9664000272750854