import os
import sys

from keras.layers import Input

from Dataset.Dataset_Utils.dataset_tools import print_cm
from Dataset.Dataset_Utils.datagen import DataGenerator
from Dataset.Dataset_Utils.augmenter import NoAug
from Models.model_sharma import SharmaNet
from inference import Inference


def _clear_past_predictions(inference):
    inference.true.clear()
    inference.predicted.clear()


class FramesClassidier:
    def __init__(self, weights_path="/user/vlongobardi/checkpoint_best.hdf5", time_step=50):
        # load model
        self.model = SharmaNet(Input(shape=(time_step, 224, 224, 3)), classification=True, weights='afew')
        self.model.load_weights(weights_path)

    def make_a_prediction(self, path):
        test_gen = DataGenerator(path, '', 1, 31, NoAug(), split_video_len=1, max_invalid=12, test=True)
        inference = Inference(model=self.model, custom_inference=True, time_step=50)
        _clear_past_predictions(inference)
        graund_truth, prediction = inference.predict_generator(test_gen, mode="overlap")
        return graund_truth, prediction
