import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', node=512, node2=256, node3=128):
    inputs = Input(shape=(x_train.shape[1]), name='input')
    x = Dense(node, activation='relu', name='hidden1')(inputs)
    x= Dropout(drop)(x)
    x = Dense(node2, activation='relu', name='hidden2')(x)
    x= Dropout(drop)(x)
    x = Dense(node3, activation='relu', name='hidden3')(x)
    x= Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    node = [128, 256, 512]
    node2 = [64, 128, 256]
    node3 = [32, 64, 128]
    epochs = [1, 2, 3]
    return {'batch_size' : batches,'optimizer' : optimizers,
            'drop' : dropout, 'node' : node, 'node2' : node2, 'node3' : node3, 'epochs' : epochs}

hyperparamters = create_hyperparameter()

model2 = KerasClassifier(build_fn=build_model, verbose=1)#, epochs=2, validation_split=0.2)

# model = RandomizedSearchCV(model2, hyperparamters, cv=5)
model = GridSearchCV(model2, hyperparamters, cv=2)

model.fit(x_train, y_train, verbose=1, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)

acc = model.score(x_test, y_test)
print('최종 스코어 : ', acc)