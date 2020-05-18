import sys,os
sys.path.append("/data/s4179447/")
import cv2
from six.moves import cPickle as pickle
from Dataset.Dataset_Utils.old_files_afew.dataset_tools import _read_dataset, findRelevantFace, equalize_hist, linear_balance_illumination, \
    mean_std_normalize, random_change_roi, random_image_rotate, random_image_skew, random_change_image, cut, roi_center, \
    show_frame, split_video, cut_centered, _random_normal_crop, cut_centered
from Dataset.Dataset_Utils.facedetect_vggface2.face_detector import FaceDetector
from tqdm import tqdm
import numpy as np
import joblib
import keras
from multiprocessing import Lock
from math import ceil
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#import Models.Config.SEResNet50_config as config
import random
from numpy import argmax
import time
import abc
#import dlib
#from Dataset.Dataset_Utils.facedetect_vggface2.face_aligner import FaceAligner


class DataGenerator(keras.utils.Sequence):
    balance_data = None
    'Generates data for Keras'

    def __init__(self, data, target_shape, with_augmentation=True, batch_size=64, custom_augmentation=None,
                 balance_data=False):
        self.mutex = Lock()
        self.data = data
        self.with_augmentation = with_augmentation
        self.custom_augmentation = custom_augmentation
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.on_epoch_end()
        self.balance_data = balance_data
        if self.balance_data: self._balance_data()

    def __len__(self):
        nitems = len(self.data)
        return ceil(nitems / self.batch_size)

    def __getitem__(self, index):
        self.mutex.acquire()
        if self.cur_index >= len(self.data):
            #print("Reset")
            self.cur_index = 0
        i = self.cur_index
        self.cur_index += self.batch_size
        self.mutex.release()
        data = self._load_batch(i)
        return data

    def on_epoch_end(self):
        self.mutex.acquire()
        self.cur_index = 0
        np.random.shuffle(self.data)
        if self.balance_data:
            self._balance_data()
        self.mutex.release()

    @abc.abstractmethod
    def _balance_data(self):
        return

    @abc.abstractmethod
    def _load_item(self, d):
        return

    def _load_frame(self, frame, augmentator):
        # refer to the increased image shape
        roi = (0, 0, frame.shape[0], frame.shape[1])
        new_shape = []
        if self.with_augmentation:
            # rotate and skew on bigger image avoiding black margin
            frame = augmentator.image_rotate(frame, roi)
            frame = augmentator.image_skew(frame, roi)
            img = augmentator.change_roi(frame)
            img = augmentator.change_image(img)
        else:
            img = cut_centered(frame, random=False)

        # Preprocess the image for the network
        img = cv2.resize(img, self.target_shape[0:2])
        img = equalize_hist(img)
        img = img.astype(np.float32)
        img = linear_balance_illumination(img)
        if np.abs(np.min(img) - np.max(img)) < 1:
            #print("WARNING: Image is =%d" % np.min(img))
        else:
            img = mean_std_normalize(img)
        if self.target_shape[2] == 3 and (len(img.shape) < 3 or img.shape[2] < 3):
            img = np.repeat(np.squeeze(img)[:, :, None], 3, axis=2)
        return img

    def _load(self, index):
        return self._load_item(self.data[index])

    def _load_batch(self, start_index, load_pairs=False):
        def get_empty_stuff(item):
            if item is None:
                return None
            stuff = []
            # stuff = [len(item)*[]]
            for j in range(len(item)):
                # np.empty( [0]+list(item[j].shape)[1:], item[j].dtype)
                stuff.append(list())
            return stuff

        item = self._load(start_index)
        stuff = get_empty_stuff(item)
        size_of_this_batch = min(self.batch_size, len(self.data) - start_index)
        for index in range(start_index, start_index + size_of_this_batch):
            if item is None:
                item = self._load(index)
            for j in range(len(item)):
                stuff[j].append(item[j])
            item = None
        for j in range(len(stuff)):
            stuff[j] = np.array(stuff[j])
            if len(stuff[j].shape) == 2 and stuff[j].shape[1] == 1:
                stuff[j] = np.reshape(stuff[j], (stuff[j].shape[0],))
        return stuff


class Augmentator():
    def __init__(self, max_change_fraction=0.008, target_shape=(224, 224)):
        # param img rotate
        self.random_angle_deg = _random_normal_crop(1, 10)[0]
        # param img skew
        self.random_skew = _random_normal_crop(2, 0.1, positive=True)
        # random img change
        self.a = _random_normal_crop(1, 0.5, mean=1)[0]
        self.b = _random_normal_crop(1, 48)[0]
        self.random_flip = random.randint(0, 1)
        # param roi change
        self.max_change_fraction = max_change_fraction
        sigma = target_shape[0] * self.max_change_fraction
        self.xy = _random_normal_crop(2, sigma, mean=-sigma / 5).astype(int)
        self.wh = _random_normal_crop(2, sigma * 2, mean=sigma / 2, positive=False).astype(int)

    def image_rotate(self, img, roi):
        return random_image_rotate(img, roi_center(roi), self.random_angle_deg)

    def image_skew(self, img, roi):
        return random_image_skew(img, roi_center(roi), self.random_skew)

    def change_image(self, img):
        return random_change_image(img, random_values=(self.a, self.b, self.random_flip))

    def change_roi(self, img):
        return cut_centered(img, random_values=(self.xy, self.wh))