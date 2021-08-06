import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
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
    '/study/_data/cat_dog/training_set',
    target_size=(200,200),
    batch_size=35,
    class_mode='binary',
    shuffle=True
)
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '/study/_data/cat_dog/test_set',
    target_size=(200,200),
    batch_size=35,
    class_mode='binary',
    shuffle=True
)
# Found 2023 images belonging to 2 classes.

#print(xy_train[0][0].shape, xy_train[0][1].shape)       # (35, 200, 200, 3) (35,)

np.save('/study/_save/_npy/k59_8_cat_dog_x_train.npy', arr=xy_train[0][0])
np.save('/study/_save/_npy/k59_8_cat_dog_y_train.npy', arr=xy_train[0][1])
np.save('/study/_save/_npy/k59_8_cat_dog_x_test.npy', arr=xy_test[0][0])
np.save('/study/_save/_npy/k59_8_cat_dog_y_test.npy', arr=xy_test[0][1])