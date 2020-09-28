import numpy as np
from keras import Model

from Dataset.Dataset_Utils.datagen import DataGenerator
from Dataset.Dataset_Utils.augmenter import NoAug
from Models.model_sharma import SharmaNet
from inference import Inference


class FramesClassifier:
    def __init__(self, weights_path="/user/vlongobardi/checkpoint_best.hdf5", time_step=16, overlap=.5):
        self.time_step = time_step
        self.model = SharmaNet((self.time_step, 224, 224, 3), classification=True, weights='afew')  # , dim=-1)
        self.model.load_weights(weights_path)
        self.feature_generator = None
        self.inference = Inference(model=self.model, custom_inference=True, time_step=self.time_step)
        self.overlap = overlap

    # #### LATE FUSION #### #
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

    # ###################### #
    # #### EARLY FUSION #### #
    def init_feature_generator(self):
        """ Define a new model to generate features for frames. Obtained removing all layers after the RNN """
        self.feature_generator = Model(self.model.input, self.model.layers[-5].output)
        # self.feature_generator.summary()

    def get_feature(self, path):
        """ Given the CSV clip, generates a List: each entry are a frame features """
        if self.feature_generator is None:
            self.init_feature_generator()

        features = []
        generator = DataGenerator(path, '', 1, 31, NoAug(), split_video_len=1, max_invalid=12, test=True)
        item = {
            'frames': generator[0][0].reshape(generator[0][0].shape[1:]),
            'label': generator[0][1]
        }
        x = item['frames']
        for i in range(0, len(item['frames'])):
            item = x[i]
            item = item[np.newaxis, np.newaxis, ...]
            features.append(self.feature_generator.predict(item))
        return features
