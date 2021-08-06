import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


""" train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip= True,
    vertical_flip= True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '/study/_data/horse-or-human',
    target_size=(200,200),
    batch_size=1030,
    class_mode='categorical',
    shuffle=True
)
# Found 1027 images belonging to 2 classes.

# xy_test = test_datagen.flow_from_directory(
#     '/study/_data/horse-or-human',
#     target_size=(150,150),
#     batch_size=5,
#     class_mode='categorical',
#     shuffle=True
# )

print(xy_train[0][0].shape, xy_train[0][1].shape)       # (1027, 200, 200, 3) (1027, 2)

np.save('/study/_save/_npy/k59_7_hores_human_x.npy', arr=xy_train[0][0])
np.save('/study/_save/_npy/k59_7_hores_human_y.npy', arr=xy_train[0][1]) """

x = np.load('/study/_save/_npy/k59_7_hores_human_x.npy')
y = np.load('/study/_save/_npy/k59_7_hores_human_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

#print(np.unique(y_train),np.unique(y_test))

#print(x_train.shape, x_test.shape)      # (2016, 200, 200, 3) (504, 200, 200, 3)

#print(x_test.shape, y_test.shape)      # (504, 200, 200, 3) (504, 3)
#print(x_train.shape, y_train.shape)      # (2016, 200, 200, 3) (2016, 3)


model = Sequential()
model.add(Conv2D(10, (2,2),padding='same', input_shape=(200,200,3), activation='relu'))
model.add(Dropout(0.9))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=20, mode='auto', verbose=1, restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.2, verbose=2, callbacks=[es])
end_time = time.time() - start_time

print('걸린시간은 : ', end_time)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

