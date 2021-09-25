import numpy as np

from tensorflow import keras
from skimage.io import imread
from skimage.transform import resize


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
