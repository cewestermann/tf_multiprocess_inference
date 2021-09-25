
"Seems to work on Windows"

import glob
import os
import time
from multiprocessing import Queue, Pool

import tensorflow as tf

from dataset import Dataset 


def get_producer():
    model = tf.keras.models.load_model('saved_models/test_model')
    def producer(fqueue, pqueue):
        while True:
            item = fqueue.get()
            print(f"Produced!")
            pred = model.predict(item) 
            pqueue.put(pred)
    return producer


def consumer(pqueue):
    while True:
        item = pqueue.get()
        print(f"Consumed! ")
        time.sleep(3)


def test():
    DATA = './data'
    
    image_paths = glob.glob(os.path.join(DATA, 'imgs/*.jpg'))
    label_paths = glob.glob(os.path.join(DATA, 'masks/*.jpg'))

    ds = Dataset(4, 224, image_paths, label_paths)

    file_queue = Queue()
    for d in ds:
        file_queue.put(d[0])

    pred_queue = Queue()


    worker_pool = Pool(4, consumer, (pred_queue,))

    producer = get_producer()
    producer(file_queue, pred_queue)
    

if __name__ == '__main__':
    test()
