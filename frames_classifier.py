import numpy as np
from keras import Input, Model

from Dataset.Dataset_Utils.datagen import DataGenerator
from Dataset.Dataset_Utils.augmenter import NoAug
from Dataset.Dataset_Utils.old_files_afew.dataset_tools import split_video
from Models.model_sharma import SharmaNet
from inference import Inference


class FramesClassifier:
    def __init__(self, weights_path="/user/vlongobardi/checkpoint_best.hdf5", time_step=50):
        self.time_step = time_step
        self.model = SharmaNet(Input(shape=(time_step, 224, 224, 3)), classification=True, weights='afew')
        self.model.load_weights(weights_path)
        self.feature_generator = None
        self.inference = Inference(model=self.model, custom_inference=True, time_step=time_step)
        self.overlap = None

    def predict(self, path):
        test_gen = DataGenerator(path, '', 1, 31, NoAug(), split_video_len=1, max_invalid=12, test=True)
        self._clear_past_predictions()
        graund_truth, prediction = self.inference.predict_generator(test_gen, mode="overlap")
        if len(graund_truth) == 0:
            print(path)
            return None, None
        return graund_truth[0], prediction[0]

    def _clear_past_predictions(self):
        self.inference.true.clear()
        self.inference.predicted.clear()

    def init_feature_generator(self, overlap=.5):
        self.feature_generator = Model(self.model.input, self.model.layers[-5].output)
        self.overlap = overlap
        # self.feature_generator.summary()

    def get_feature(self, path):
        if self.feature_generator is None:
            self.init_feature_generator()

        features = []
        generator = DataGenerator(path, '', 1, 31, NoAug(), split_video_len=1, max_invalid=12, test=True)
        item = {
            'frames': generator[0][0].reshape(generator[0][0].shape[1:]),
            'label': generator[0][1]
        }

        x = item['frames']
        if len(item['frames']) < self.model.input_shape[1]:
            x = split_video(item=item, split_len=self.model.input_shape[1])[0]['frames']

        for i in range(0, (len(x) - self.time_step + 1), int(self.time_step * self.overlap)):
            item = x[i:i + self.time_step]
            item = item[np.newaxis, ...]
            features.append(self.feature_generator.predict(item))
        return features
