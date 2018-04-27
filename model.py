import os

from keras.layers import Conv2D, MaxPooling2D, Dense, \
    Flatten, Dropout
from keras.models import Sequential, load_model

saved_dir = './saved'
model_path = 'best_model.h5'
model_history_path = 'best_model_history'


def get_model(num_classes, input_shape):
    # Load saved model if exists
    if os.path.exists(model_path):
        print("-------------------Load model---------------------------")
        model = load_model(model_path)
    else:
        print("-------------------Create model---------------------------")
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                         input_shape=input_shape))
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

        model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model
