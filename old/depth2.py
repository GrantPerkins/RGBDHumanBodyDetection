import struct
from random import randint
import tensorflow as tf
import numpy as np
from os import listdir
import math
from time import time


class Dataset:
    def __init__(self, length, width):
        start = time()
        self.length = length
        self.width = width
        filenames = ["./depthframes/" + f for f in listdir("./depthframes/")]
        self.images = len(filenames)
        self._frames = []
        for i, file in enumerate(filenames):
            if i % 70 == 0:
                print("%.1f%% images loaded" % ((i / self.images) * 100))
            img = DepthImage(file, self.length, self.width)
            img.random_crop()
            self._frames.append(img.frame.tolist())
        self.i = 0
        print(time() - start)

    def restart(self):
        self.i = 0

    def next_batch(self, batch_size=10):
        self.i += 1
        return self._frames[(self.i - 1) * batch_size:self.i * batch_size]


class DepthImage:
    def __init__(self, path, length, width):
        self.rows, self.cols = 240, 320
        self.length, self.width = length, width
        self.f = open(path, 'rb')
        self.frame = []
        self.get_frame()

    def _get_frame(self):
        for row in range(self.rows):
            tmp_row = []
            for col in range(self.cols):
                tmp_row.append([struct.unpack('i', self.f.read(4))[0], ])
            tmp_row = [[0, ] if math.isnan(i[0]) else list(map(int, i)) for i in tmp_row]
            self.frame.append(tmp_row)

    def get_frame(self):
        """
        Gets header, and first frame
        :return:
        """
        self._get_header()
        self._get_frame()

    def random_crop(self):
        x, y = randint(0, 239 - self.length), randint(0, 319 - self.width)
        frame = np.array(self.frame)[x:x + self.length, y:y + self.width]
        self.frame = frame.reshape(self.length * self.width)

    def get_frame(self):
        self._get_frame()
        self.frame = np.array(self.frame).reshape(240, 320)
