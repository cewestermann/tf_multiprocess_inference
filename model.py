'''
NOTES:
    Look at tf.keras.utils.Sequence for multiprocessing compatibility
'''

import os
import glob

import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

from dataset import Dataset

#tf.debugging.set_log_device_placement(True)

DATA = './data'

image_paths = glob.glob(os.path.join(DATA, 'imgs/*.jpg'))
label_paths = glob.glob(os.path.join(DATA, 'masks/*.jpg'))

image_tests = [image_paths.pop() for _ in range(5)]
label_tests = [label_paths.pop() for _ in range(5)]


def get_model(img_size):
    inputs = keras.Input(shape=img_size)
    x = keras.layers.Conv2D(8, 3, padding="same")(inputs)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(16, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    outputs = keras.layers.Activation("relu")(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
            image_paths, label_paths, test_size=0.10, random_state=123)

    ds_train = Dataset(4, 224, X_train, y_train)
    ds_val = Dataset(4, 224, X_test, y_test)
    ds_test = Dataset(4, 224, image_tests, label_tests)

    model = get_model((224, 224, 3))
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    epochs = 5
    model.fit(ds_train, epochs=epochs, validation_data = ds_val)
    model.save('saved_models/test_model')
