import os
import pickle
from operator import itemgetter

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10
from keras.models import clone_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

from model import set_model_params, saved_dir, create_model, load_saved_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

IMAGE_SIZE, CHANNELS = x_train.shape[2:]
print(IMAGE_SIZE, CHANNELS)
num_classes = len(np.unique(y_train))
print('num_classes = ' + str(num_classes))

print("Training set size:\t", len(y_train))
print("Testing set size:\t", len(y_test))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

DEFAULT_MODEL_PATH = 'best_model.h5'
DEFAULT_MODEL_HISTORY_PATH = 'best_model_history'
DEF_TRAN_PATH = 'transition_model.h5'
DEF_TRAN_HIS_PATH = 'transition_model_history'

DEF_FINAL_PATH = 'final_model.h5'

set_model_params(num_classes, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))


def train(_model, epochs, patience, saved_path=None):
    _model = compile_model(_model)
    cb_list = []
    if saved_path is not None:
        checkpoint = ModelCheckpoint(saved_dir + '/' + saved_path,
                                     # model filename
                                     monitor='val_loss',  # quantity to monitor
                                     verbose=0,  # verbosity - 0 or 1
                                     save_best_only=True,
                                     # The latest best model will not be
                                     # overwritten
                                     # The decision to overwrite model
                                     mode='auto')
        cb_list.append(checkpoint)

    early_stopping = EarlyStopping(monitor="val_loss", patience=patience,
                                   verbose=1, mode='auto')
    cb_list.append(early_stopping)

    _model_details = _model.fit(x_train, y_train,
                                batch_size=32,
                                # number of samples per gradient update
                                epochs=epochs,  # number of iterations
                                validation_data=(x_test, y_test),
                                callbacks=cb_list,
                                verbose=2)
    return _model, _model_details


def compile_model(_model):
    _model.compile(loss='sparse_categorical_crossentropy',
                   optimizer=Adam(lr=1.0e-4),
                   # Adam optimizer with 1.0e-4 learning rate
                   # Metrics to be evaluated by the model
                   metrics=['accuracy'])
    return _model


if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)


# Save model history
def save_history(_model_details, _history_path):
    with open(saved_dir + '/' + _history_path, 'w+b') as file_pi:
        pickle.dump(_model_details.history, file_pi)


model = load_saved_model(DEFAULT_MODEL_PATH)
if model is None:
    model = create_model()

# Train default model
model, model_details = train(model, 100, 10, saved_path=DEFAULT_MODEL_PATH,)

save_history(model_details, DEFAULT_MODEL_HISTORY_PATH)

# Evaluate the model
score_ori = model.evaluate(x_test, y_test, verbose=0)
print("Before Pruning, accuracy: %.2f%%" % (score_ori[1] * 100) + '\n')

ori_weights = model.get_weights()
print("{} of {} is zero".format(np.count_nonzero(ori_weights[0] == 0),
                                ori_weights[0].size))

# Predictions
class_pred = model.predict(x_test, batch_size=32)

# Predict class for test set gesture
labels_pred = np.argmax(class_pred, axis=1)

# Print metrics
cm = confusion_matrix(y_true=y_test[:, 0], y_pred=labels_pred)
row_format = '{:>5}' * (len(class_names) + 1)
print(row_format.format(*class_names, " "))
for label, row, i in zip(class_names, cm, range(len(class_names))):
    print(row_format.format(*row, " ({}){}".format(i, label)))

print("\n---------------Start Pruning----------------------------------")

# layers need to prune:
# 1,2,5,6,9,10,14,15
prune_layouts = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5',
                 'conv2d_6', 'dense_1', 'dense_2']

prune_thresholds = [[0.01, 0.06, 0.005], [0.01, 0.1, 0.01], [0.01, 0.1, 0.01],
                    [0.01, 0.1, 0.01], [0.01, 0.1, 0.01], [0.01, 0.1, 0.01],
                    [0.01, 0.1, 0.01], [0.01, 0.1, 0.01]]


# prune function
def prune(matrix, _threshold):
    useless_idx = (np.absolute(matrix) < _threshold)
    matrix[useless_idx] = 0
    return matrix


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += jump


optimal_sol = {}

for layer_name, thresholds in zip(prune_layouts, prune_thresholds):
    print('Pruning layer ' + layer_name)
    model_copy = clone_model(model)
    model_copy.set_weights(model.get_weights())
    # Generate layer dictionary from cloned model
    layer_dict = dict([(layer.name, layer) for layer in model_copy.layers])
    layer = layer_dict[layer_name]
    weights = layer.get_weights()
    accuracy_threshold = []
    for threshold in drange(thresholds[0],
                            thresholds[1],
                            thresholds[2]):
        # weights[0] is weight tensor, weights[1] is bias tensor,
        # prune weights only
        print('Current threshold is {:.3f}'.format(threshold))
        saved_name = layer_name + '_{:.3f}'.format(threshold)
        tmp_model = load_saved_model(saved_name + '.h5')
        if tmp_model is not None:
            print('Already trained')
            model_copy = compile_model(tmp_model)
        else:
            weights_copy = np.copy(weights)
            weights_copy[0] = prune(weights_copy[0], threshold)
            layer.set_weights(weights_copy)
            print("sparsity:",
                  np.count_nonzero(weights_copy[0]) / weights_copy[0].size, '\n')
            model_copy, model_copy_detail = train(model_copy, 50, 5)
            model_copy.save(filepath=saved_name + '.h5')
            save_history(model_copy_detail, saved_name + '_history')
        score_copy = model_copy.evaluate(x_test, y_test, verbose=0)
        print("After Pruning, accuracy: %.2f%%" % (score_copy[1] * 100) + '\n')
        accuracy_threshold.append((threshold, score_copy[1]))
    best_threshold, best_threshold_acc = max(accuracy_threshold,
                                             key=itemgetter(1))
    optimal_sol[layer_name] = best_threshold
    print("Layout {}, Best threshold: {:.3f}, accuracy: {:.2%} \n"
          .format(layer_name, best_threshold, best_threshold_acc))

print("\nOur optimal_sol")
for k in optimal_sol:
    print(k, ':', optimal_sol[k])

print("\n---------------Combine Pruning Training-----------------------------")

tmp_model = load_saved_model(DEF_TRAN_PATH)
if tmp_model is not None:
    print('Already trained')
    model_transition = compile_model(tmp_model)
else:
    model_transition = clone_model(model)
    model_transition.set_weights(model.get_weights())
    # Generate layer dictionary from cloned model
    layer_dict = dict([(layer.name, layer) for layer in model_transition.layers])
    for layer_name in prune_layouts:
        t_layer = layer_dict[layer_name]
        t_weights = t_layer.get_weights()
        t_threshold = optimal_sol[layer_name]
        t_weights[0] = prune(t_weights[0], t_threshold)
        t_layer.set_weights(t_weights)
        print("Layout: {}, threshold: {:.3f},  sparsity: {}"
              .format(layer_name, t_threshold,
                      np.count_nonzero(t_weights[0]) / t_weights[0].size))
    # Train one time
    model_transition, model_transition_detail = train(model_transition, 100, 10)
    model_transition.save(DEF_TRAN_PATH)
    save_history(model_transition_detail, DEF_TRAN_HIS_PATH)

score_transition = model_transition.evaluate(x_test, y_test, verbose=0)
print("After combine pruning, accuracy: %.2f%%" % (
        score_transition[1] * 100) + '\n')

print("\n---------------Final Pruning Training -----------------------------")
tmp_model = load_saved_model(DEF_FINAL_PATH)
if tmp_model is not None:
    print('Already trained')
    model_final = compile_model(tmp_model)
else:
    model_final = clone_model(model_transition)
    model_final.set_weights(model_transition.get_weights())
    # Generate layer dictionary from cloned model
    layer_dict = dict([(layer.name, layer) for layer in model_final.layers])
    for layer_name in prune_layouts:
        f_layer = layer_dict[layer_name]
        f_weights = f_layer.get_weights()
        f_threshold = optimal_sol[layer_name]
        f_weights[0] = prune(f_weights[0], f_threshold)
        f_layer.set_weights(f_weights)
        print("Layout {}, threshold: {:.3f}, sparsity: {}"
              .format(layer_name, f_threshold,
                      np.count_nonzero(f_weights[0]) / f_weights[0].size))
    model_final = compile_model(model_final)

score_final = model_final.evaluate(x_test, y_test, verbose=0)
print("After combine pruning, accuracy: %.2f%%" % (
        score_final[1] * 100) + '\n')
model_final.save(DEF_FINAL_PATH)

final_weights = model_final.get_weights()
print("{} of {} is zero".format(np.count_nonzero(final_weights[0] == 0),
                                final_weights[0].size))

