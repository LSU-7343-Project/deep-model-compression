import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, \
    Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

IMAGE_SIZE, CHANNELS = x_train.shape[2:]
print(IMAGE_SIZE, CHANNELS)
print(np.unique(y_train))
num_classes = len(np.unique(y_train))

print("Training set size:\t", len(y_train))
print("Testing set size:\t", len(y_test))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']


def vgg_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
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


model = vgg_model()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=1.0e-4),
              # Adam optimizer with 1.0e-4 learning rate
              metrics=['accuracy'])  # Metrics to be evaluated by the model

checkpoint = ModelCheckpoint('best_model.h5',  # model filename
                             monitor='val_loss',  # quantity to monitor
                             verbose=0,  # verbosity - 0 or 1
                             save_best_only=True,
                             # The latest best model will not be overwritten
                             mode='auto')  # The decision to overwrite model

model_details = model.fit(x_train, y_train,
                          batch_size=128,
                          # number of samples per gradient update
                          epochs=100,  # number of iterations
                          validation_data=(x_test, y_test),
                          callbacks=[checkpoint],
                          verbose=1)

model.save_weights('cifar10vgg.h5')

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

class_pred = model.predict(x_test, batch_size=32)
print(class_pred[0])

labels_pred = np.argmax(class_pred, axis=1)
print(labels_pred)

correct = (labels_pred == y_test[:, 0]).astype('int32')
print(correct)
print("Number of correct predictions: %d" % sum(correct))

num_images = len(correct)
print("Accuracy: %.2f%%" % ((sum(correct) * 100) / num_images))
