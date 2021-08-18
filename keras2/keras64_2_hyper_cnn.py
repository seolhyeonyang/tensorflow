# 실습
#TODO CNN 변경 / 파라미터 변경 (노드 개수, activation, epochs=[1, 2, 3], learning_rate)

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, GlobalAveragePooling2D
from tensorflow.python.keras.backend import dropout


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def build_model(drop=0.5, optimizer='adam', node=100, node2=200):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(20, (2, 2), padding='same', name='hidden1')(inputs)
    x= Dropout(drop)(x)
    x = Dense(10, (2,2), activation='relu', name='hidden2')(x)
    x = GlobalAveragePooling2D()(x)
    x= Dropout(drop)(x)
    x = Dense(node, activation='relu', name='hidden3')(x)
    x= Dropout(drop)(x)
    x = Dense(node2, activation='relu', name='hidden4')(x)
    x= Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='categorical_crossentropy')
    return model

from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    adam = Adam(learning_rate=[0.1, 0.01, 0.001, 0.001])
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    node = [128, 256, 512]
    node2 = [64, 128, 256]
    return {'batch_size' : batches,'optimizer' : optimizers,
            'drop' : dropout, 'node' : node, 'node2' : node2} 

hyperparamters = create_hyperparameter()
# print(hyperparamters)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# model2 = build_model()
model2 = KerasClassifier(build_fn=build_model, verbose=1)#, epochs=2, validation_split=0.2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# model = RandomizedSearchCV(model2, hyperparamters, cv=5)
model = GridSearchCV(model2, hyperparamters, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
'''
{'batch_size': 1000, 'drop': 0.1, 'optimizer': 'rmsprop'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000027D607EDB80>
0.9177166819572449
'''
acc = model.score(x_test, y_test)
print('최종 스코어 : ', acc)
# 최종 스코어 :  0.9634000062942505