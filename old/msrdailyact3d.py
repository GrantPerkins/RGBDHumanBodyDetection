import tensorflow as tf
import numpy as np
import struct
import math
from os import listdir
from random import randint


class Dataset:
    def __init__(self, sae=True):
        """
        Creates a tf.Dataset iterator that returns batches of images.
        Advantagous over preloading all images because it uses less RAM. It must crop each image at runtime, however.
        :param sae:
        """
        # if sae is true, random cropping is done
        self.sae = sae

        filenames = ["./depthframes/" + f for f in listdir("./depthframes/")]
        self._dataset = tf.data.Dataset.from_tensor_slices(filenames) \
            .map(lambda name: tf.py_func(self._parse, [name], tf.int32)) \
            .repeat()

        self._batched = self._dataset.batch(10)
        self._iterator = self._batched.make_one_shot_iterator()
        print("Dataset constructed")

    def restart(self):
        """
        Creates new iterator, useful when old iterator is spent
        """
        self._iterator = self._batched.make_one_shot_iterator()

    def next(self):
        """
        Gets next batch
        :return: the next batch in the iterator
        """
        return self._iterator.get_next()

    def _parse(self, filename):
        """
        Gets filename
        :param filename:
        :return:
        """
        img = DepthImage(filename)
        if self.sae:
            img.random_crop()
        return img.frame


class DepthImage:
    def __init__(self, path):
        """
        Constructs attributes of image
        :param path: path to depth image
        """
        self.rows, self.cols = 240, 320
        self.length = self.width = 40
        self.f = open(path, 'rb')
        self.frame = []
        self.get_frame()

    def _get_frame(self):
        """
        Sets self.frame to a 3d list that is the image
        """
        for row in range(self.rows):
            tmp_row = []
            for col in range(self.cols):
                tmp_row.append([struct.unpack('i', self.f.read(4))[0], ])
            tmp_row = [[0, ] if math.isnan(i[0]) else list(map(int, i)) for i in tmp_row]
            self.frame.append(tmp_row)

    def random_crop(self):
        """
        Crops a random 40x40 square out of the full image
        :return:
        """
        x, y = randint(0, 239 - self.length), randint(0, 319 - self.width)
        frame = np.array(self.frame)[x:x + self.length, y:y + self.width]
        self.frame = frame.reshape(40 * 40)

    def get_frame(self):
        """
        Takes the frame from self._get_frame, reshapes it to 240x320
        :return:
        """
        self._get_frame()
        self.frame = np.array(self.frame).reshape(240, 320)


if __name__ == "__main__":
    d = Dataset()
