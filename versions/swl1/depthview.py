"""
Copyright (c) 2018 Grant Perkins
"""

import numpy as np
import struct
from time import time
import math
from random import randint


class Dataset:
    """
    Generates a 4d list of depth images.
    """

    def __init__(self, crop=True, length=40, width=40):
        """
        Sets values, including path to videos
        calls self._generate_videos
        :param length:
        :param width:
        """
        self.length, self.width = length, width
        self.videos = None
        self.frames = []
        self.uncropped = []
        self.path = "C:\\Users\\gcper\\Code\\STEM\\data\\MSRDailyAct3D_pack1\\"
        self.template = "a{}_s{}_e{}_depth.bin"
        self._generate_videos(crop)
        self.i = 0
        self.images = 0

    def _generate_videos(self, crop):
        """
        Reads all video files, generates full list
        """
        print("Loading dataset")
        self.frames = []
        start = time()
        a_, s_, e_ = [6, 10, 1]
        for a in range(a_):  # out of 2
            for s in range(s_):  # out of 10
                for e in range(e_):  # out of 1 (value = 2)
                    percent = (a / a_) + (s / (a_ * s_)) + (e / (a_ * s_ * e_))
                    print("%.1f%% images loaded" % (((a / a_) + (s / (a_ * s_)) + (e / (a_ * s_ * e_))) * 100))
                    filename = self.template.format(*[str(i + 1).zfill(2) for i in [a, s, 1]])
                    print(filename)
                    tmp_video = DepthVideo(self.path + filename, self.length, self.width, crop)
                    tmp_video.get_frames()
                    print(round(((time() - start) / ((percent * 60) + 1e-10)) - ((time() - start) / 60)), "minutes remaining")
                    self.frames.extend(tmp_video.frames)
                    self.uncropped.extend(tmp_video.uncropped)
        print("100% images loaded")
        print("Elapsed time: {}".format(time() - start))
        print("{} frames available for training".format(len(self.frames)))
        self.images = len(self.frames)

    def restart(self):
        """
        Restarts iterator
        """
        self.i = 0

    def next_batch(self, batch_size, images):
        """
        Gets the next batch from given images
        :param batch_size: size of batch desired
        :param images: list of images
        :return: next batch
        """
        self.i += 1
        return images[(self.i - 1) * batch_size:self.i * batch_size]


class DepthVideo:
    def __init__(self, path, length, width, crop):
        """
        Sets values of video
        :param path:
        :param length:
        :param width:
        """
        self.length, self.width = length, width
        self.f = open(path, 'rb')
        self.nb_frames, self.nb_cols, self.nb_rows = 0, 0, 0
        self.frames = []
        self.i = 0
        self.saved_real = False
        self.frame = []
        self.uncropped = []
        self.crop = crop

    def get_frame(self):
        self._get_header()
        self._get_frame()

    def _get_header(self):
        """
        Gets the header values from the file
        """
        self.nb_frames = struct.unpack('i', self.f.read(4))[0]
        self.nb_cols = struct.unpack('i', self.f.read(4))[0]
        self.nb_rows = struct.unpack('i', self.f.read(4))[0]

    def _get_frame(self):
        """
        Reads the next frame from the video file
        """
        frame = []
        for row in range(self.nb_rows):
            tmp_row = []
            for col in range(self.nb_cols):
                tmp_row.append([struct.unpack('i', self.f.read(4))[0], ])
            tmp_row = [[0, ] if math.isnan(i[0]) else list(map(int, i)) for i in tmp_row]
            self.f.read(self.nb_cols)
            frame.append(tmp_row)
        uncropped = frame
        if self.crop:
            # crop random 40 by 40 image
            if not self.saved_real:
                self.frame = np.array(frame)
                self.saved_real = True
            x, y = randint(0, 239 - self.length), randint(0, 319 - self.width)
            frame = np.array(frame)[x:x + self.length, y:y + self.width]
            return uncropped, frame.tolist()
        return uncropped, uncropped

    def get_frames(self):
        """
        Gets header, and all frames from video
        """
        self._get_header()
        self.frames = []
        for i in range(self.nb_frames):
            uncropped, frame = self._get_frame()
            self.frames.append(frame)
            self.uncropped.append(uncropped)
