import os

from keras.layers import Conv2D, MaxPooling2D, Dense, \
    Flatten, Dropout
from keras.models import Sequential, load_model

saved_dir = './saved'
# model_path = 'best_model.h5'
# model_history_path = 'best_model_history'

_num_classes = 0
_input_shape = None


def set_model_params(num_classes, input_shape):
    global _num_classes
    global _input_shape
    _num_classes = num_classes
    _input_shape = input_shape


def create_model():
    print("-------------------Create model---------------------------")
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=_input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(_num_classes, activation='softmax'))
    model.summary()
    return model


def load_saved_model(model_path):
    model = None
    model_path = saved_dir + '/' + model_path
    if os.path.exists(model_path):
        # Load saved model if exists
        # print("-------------------Load model---------------------------")
        model = load_model(model_path)
    return model
