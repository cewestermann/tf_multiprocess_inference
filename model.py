'''
NOTES:
    Look at tf.keras.utils.Sequence for multiprocessing compatibility
'''

import os
import glob

import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from skimage.io import imread
from skimage.transform import resize

DATA = './data'

image_paths = glob.glob(os.path.join(DATA, 'imgs/*.jpg'))
label_paths = glob.glob(os.path.join(DATA, 'masks/*.jpg'))

image_tests = [image_paths.pop() for _ in range(5)]
label_tests = [label_paths.pop() for _ in range(5)]


class Dataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, img_paths, target_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):

        def preprocess_batch(batch):
            return np.array([resize(imread(filename), (self.img_size, self.img_size))
                for filename in batch])

        batch_x = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.target_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        return preprocess_batch(batch_x), preprocess_batch(batch_y)


def get_model(img_size):
    inputs = keras.Input(shape=img_size)
    x = keras.layers.Conv2D(8, 3, padding="same")(inputs)
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




