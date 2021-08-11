import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from datetime import datetime
from tensorflow.keras.optimizers import Adam



train      = pd.read_csv('/study/dacon/news/_data/train_data.csv')
test       = pd.read_csv('/study/dacon/news/_data/test_data.csv')
submission = pd.read_csv('/study/dacon/news/_save/submission.csv')
topic_dict = pd.read_csv('/study/dacon/news/_data/topic_dict.csv')

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()
train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=30000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')

#print(train_tf_text.shape, test_tf_text.shape)      #(45654, 150000) (9131, 150000)

def dnn_model():
    model = Sequential()
    model.add(Dense(500, input_dim = 30000, activation = "relu"))
    model.add(Dropout(0.8))
    model.add(Dense(7, activation = "softmax",
                    kernel_regularizer=regularizers.l2(0.001),
                    kernel_initializer=tf.initializers.random_normal(0.0,1.0,seed=1)))
    return model

model = dnn_model()

logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)
# tensorboard --logdir=./logs/fit/

###################################################
date = datetime.now()
date_time = date.strftime('%m%d_%H%M')

filepath = '/study2/dacon/news/_save/'
filename = '{epoch:04d}_{val_accuracy:.4f}.hdf5'
modelpath = ''.join([filepath, 'YSH11_', date_time, "-", filename])
###################################################

optimizer = Adam(lr=0.001)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

es = EarlyStopping(monitor= 'val_accuracy', patience=30, mode='auto', verbose=2, restore_best_weights=True)

cp = ModelCheckpoint(monitor='val_accuracy', save_best_only=True, mode='auto',
                    filepath= modelpath)

reduce_lr = ReduceLROnPlateau(monitor= 'val_accuracy', patience=5, mode='auto', verbose=1, factor=0.05)


history = model.fit(x = train_tf_text[:40000], y = train_label[:40000],
                    validation_data =(train_tf_text[40000:], train_label[40000:]),
                    epochs = 2000, callbacks=[es, cp, tensorboard_callback, reduce_lr], verbose=2)

model.save('/study2/dacon/news/_save/YSH11.h5')

# print('=================== load_model ===================')
# model = load_model('/study2/dacon/news/_save/YSH5_0809_1203-0005_0.7890.hdf5')

results = model.evaluate(train_tf_text, train_label)
# print(results)
print("accuracy : ", results[1])

temp = model.predict(test_tf_text)

temp = tf.argmax(temp, axis=1)

temp = pd.DataFrame(temp)

temp.rename( columns={0:'topic_idx'}, inplace=True )

temp['index'] = np.array(range(45654,45654+9131))

temp = temp.set_index('index')


temp.to_csv('/study2/dacon/news/_save/submission11.csv')

'''
YSH5
accuracy :  0.9524028301239014

YSH6
accuracy :  0.876155436038971
'''