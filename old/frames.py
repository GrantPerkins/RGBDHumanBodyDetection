"""
Copyright (c) 2018 Grant Perkins
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from time import time
import math
from random import randint


class Generator:
    """
    Generates binary files, each being a frame from an MSRDailyActivity3D video. FOr use with Tensorflow's
    Dataset API.
    """

    def __init__(self):
        self.path = "C:\\Users\\gcper\\Code\\STEM\\data\\MSRDailyAct3D_pack1\\"
        self.template = "a{}_s{}_e{}_depth.bin"
        self._generate_videos()

    def _generate_videos(self):
        print("Loading dataset")
        start = time()
        """
        file names are in the format a{}_s{}_e{}_depth.bin
        a, s, e values range from 1-2, 1-10, and 1-2 respectedly
        """
        for a in range(2):  # out of 2
            for s in range(10):  # out of 10
                for e in range(2):  # out of 2
                    # output percent complete generating images, this takes forever
                    print("%.1f%% complete" % (((a / 2) + (s / 20) + (e / 40)) * 100))
                    # generate filename
                    filename = self.template.format(*[str(i + 1).zfill(2) for i in [a, s, e]])
                    tmp_video = DepthVideo(self.path + filename, [a, s, e])
                    tmp_video.generate_frames()
        print("100% complete")
        print("Elapsed time: {}".format(time() - start))


class DepthVideo:
    def __init__(self, path, ase):
        self.ase = ase
        self.f = open(path, 'rb')
        self.nb_frames, self.nb_cols, self.nb_rows = 0, 0, 0

    def _get_header(self):
        self.header = [self.f.read(4), self.f.read(4), self.f.read(4)]
        self.nb_frames, self.nb_cols, self.nb_rows = [struct.unpack('i', i)[0] for i in self.header]

    def _write_frame(self, frame_nb):
        path = "C:\\Users\\gcper\\Code\\STEM\\depthframes\\"
        name = "{}_{}_{}_{}.bin".format(*self.ase, frame_nb)
        with open(path + name, "wb") as file:
            # for value in self.header:
            #     file.write(value)
            for row in range(self.nb_rows):
                for col in range(self.nb_cols):
                    file.write(self.f.read(4))
                self.f.read(self.nb_cols)

    def generate_frames(self):
        self._get_header()
        for i in range(self.nb_frames):
            self._write_frame(i)


Generator()