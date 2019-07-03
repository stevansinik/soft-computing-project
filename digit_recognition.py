#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:46:57 2019

@author: student
"""

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

import parameters

def build_model():
    model = Sequential()
    
    model.add(Dense(parameters.number_of_hidden_neurons, input_shape=(parameters.mnist_digit_area,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(parameters.dropout_rate))
    
    model.add(Dense(parameters.number_of_hidden_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(parameters.dropout_rate))
    
    model.add(Dense(parameters.number_of_classes))
    model.add(Activation('softmax'))
    
    return model

def write_model(model):
    model_json = model.to_json()
    with open('ann_structure_1.json', 'w+') as file:
        file.write(model_json)
        file.flush()
    model.save_weights("ann_weights_1.h5")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

x_train = X_train.reshape(X_train.shape[0], parameters.mnist_digit_area)
x_test = X_test.reshape(X_test.shape[0], parameters.mnist_digit_area)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, parameters.number_of_classes)
y_test = np_utils.to_categorical(y_test, parameters.number_of_classes)

model = build_model()
model.compile(
        optimizer=parameters.optimizer, 
        loss=parameters.loss, 
        metrics=parameters.metrics)
history = model.fit(
        x_train, 
        y_train, 
        batch_size=parameters.batch_size, 
        epochs=parameters.epochs, 
        validation_split=parameters.validation_split)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

write_model(model)