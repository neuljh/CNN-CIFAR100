

import tensorflow as tf

from tensorflow import keras
import numpy as np 
from keras.datasets import cifar100
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
#
#
# class PlotLossAccuracyCallback(tf.keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.accuracies = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.accuracies.append(logs.get('accuracy'))
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(self.losses)
#         plt.title('Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.subplot(1, 2, 2)
#         plt.plot(self.accuracies)
#         plt.title('Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.show()


(x_train, y_train),(x_test, y_test) = cifar100.load_data(label_mode='fine')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

#y_train = keras.utils.to_categorical(y_train, 10)
#y_test = keras.utils.to_categorical(y_test, 10)

# model = keras.Sequential([
#     keras.layers.Conv2D(64, (3, 3), strides=1, input_shape=(32, 32, 3)),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.Conv2D(64, (3, 3), strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.Conv2D(128, (3, 3), strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.Conv2D(128, (3, 3), strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     # keras.layers.Dropout(0.4),
#     keras.layers.Conv2D(256, (3, 3), strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.Conv2D(256, (3, 3), strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), strides=1),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64*8*8, activation='relu'),
#     keras.layers.Dense(100, activation='softmax')
# ])

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), strides=1, input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(64, (3, 3), strides=1),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(128, (3, 3), strides=1),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(64, (3, 3), strides=1),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(64*8*8, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='softmax')
])

learning_rate = 0.002
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
# history = model.fit(x_train, y_train, epochs=10, callbacks=[PlotLossAccuracyCallback()])

import matplotlib.pyplot as plt

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test acc: %f' % test_acc)

# fig, ax = plt.subplots(figsize=(10,6))
# gen = ax.plot(plt.history.history)

