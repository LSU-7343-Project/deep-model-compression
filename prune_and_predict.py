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

model.load_weights('cifar10vgg.h5')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=1.0e-4),
              # Adam optimizer with 1.0e-4 learning rate
              metrics=['accuracy'])  # Metrics to be evaluated by the model

scores = model.evaluate(x_test, y_test, verbose=0)
#print("Accuracy before pruning: %.2f%%" % (scores[1] * 100))

class_pred = model.predict(x_test, batch_size=32)
#print(class_pred[0])

labels_pred = np.argmax(class_pred, axis=1)
#print(labels_pred)

correct = (labels_pred == y_test[:, 0]).astype('int32')
#print(correct)

print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
print("Number of correct predictions before pruning: %d" % sum(correct))

num_images = len(correct)
print("Accuracy before pruning: %.2f%%" % ((sum(correct) * 100) / num_images))
print('\n')

'''
    pruning
'''
#prune function
def prune(matrix,threshold):
    unuseful_idx = (np.absolute(matrix) < threshold)
    matrix[unuseful_idx] = 0
    return matrix
   
# layers need to prune:
# 1,2,5,6,9,10,14,15

print('pruning conv2d_1 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['conv2d_1'].get_weights()
threshold = 0 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['conv2d_1'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')

print('pruning conv2d_2 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['conv2d_2'].get_weights()
threshold = 0.015 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['conv2d_2'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')

print('pruning conv2d_3 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['conv2d_3'].get_weights()
threshold = 0.015 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['conv2d_3'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')

print('pruning conv2d_4 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['conv2d_4'].get_weights()
threshold = 0 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['conv2d_4'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')

print('pruning conv2d_5 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['conv2d_5'].get_weights()
threshold = 0.02 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['conv2d_5'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')

print('pruning conv2d_6 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['conv2d_6'].get_weights()
threshold = 0.02 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['conv2d_6'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')


print('pruning dense_1 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['dense_1'].get_weights()
threshold = 0.05 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['dense_1'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')


print('pruning dense_2 layer...')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
params = layer_dict['dense_2'].get_weights()
threshold = 0.05 # prune the model according to the threshold
                 # params[0] is weight tensor, params[1] is bias tensor, prune weights only
params[0] = prune(params[0],threshold)
layer_dict['dense_2'].set_weights(params)
print("sparsity:", np.count_nonzero(params[0])/params[0].size,'\n')

'''
    predict using pruned model
'''
scores = model.evaluate(x_test, y_test, verbose=0)
#print("Accuracy before pruning: %.2f%%" % (scores[1] * 100))

class_pred = model.predict(x_test, batch_size=32)
#print(class_pred[0])

labels_pred = np.argmax(class_pred, axis=1)
#print(labels_pred)

correct = (labels_pred == y_test[:, 0]).astype('int32')
#print(correct)

print("Number of correct predictions after pruning: %d" % sum(correct))

num_images = len(correct)
print("Accuracy after pruning: %.2f%%" % ((sum(correct) * 100) / num_images))



