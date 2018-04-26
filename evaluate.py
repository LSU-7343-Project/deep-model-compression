import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, \
    Flatten, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model_before = load_model("before_finetune.h5")
model_after = load_model("after_finetune.h5")

# model.load_weights('cifar10vgg.h5')

#model.compile(loss='sparse_categorical_crossentropy',
#              optimizer=Adam(lr=1.0e-4),
#              # Adam optimizer with 1.0e-4 learning rate
#              metrics=['accuracy'])  # Metrics to be evaluated by the model

scores = model_before.evaluate(x_test, y_test, verbose=0)
#print("Accuracy before pruning: %.2f%%" % (scores[1] * 100))

class_pred = model_before.predict(x_test, batch_size=32)
#print(class_pred[0])

labels_pred = np.argmax(class_pred, axis=1)
#print(labels_pred)

correct = (labels_pred == y_test[:, 0]).astype('int32')
#print(correct)

num_images = len(correct)
print("Accuracy before pruning: %.2f%%" % ((sum(correct) * 100) / num_images))
print('\n')
print('---------------------------------------------------------------')
scores = model_after.evaluate(x_test, y_test, verbose=0)
#print("Accuracy before pruning: %.2f%%" % (scores[1] * 100))

class_pred = model_after.predict(x_test, batch_size=32)
#print(class_pred[0])

labels_pred = np.argmax(class_pred, axis=1)
#print(labels_pred)

correct = (labels_pred == y_test[:, 0]).astype('int32')
#print(correct)

num_images = len(correct)
print("Accuracy after pruning: %.2f%%" % ((sum(correct) * 100) / num_images))
print('\n')

print('pruning dense_2 layer...')
layer_dict = dict([(layer.name, layer) for layer in model_before.layers])
params = layer_dict['dense_2'].get_weights()
print(params)
print('\n')
print('--------------------------------------------------------------')
print('\n')
layer_dict = dict([(layer.name, layer) for layer in model_after.layers])
params = layer_dict['dense_2'].get_weights()
print(params)
