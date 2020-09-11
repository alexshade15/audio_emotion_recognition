import sys
import glob
import random
import operator
import librosa
import numpy as np
from math import floor

from keras.models import load_model
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm

from keras_yamnet.yamnet import YAMNet
from keras_yamnet.preprocessing import preprocess_input

# from abstract_audio_classifier import AudioClassifier

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],


def augment(spec, num_mask=2, freq_masking_max_percentage=0.1):
    spec = spec.copy()
    ow = np.min(spec)
    for i in range(num_mask):
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(round(freq_percentage * spec.shape[1]))
        f0 = np.random.uniform(low=0.0, high=spec.shape[1] - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = ow
        time_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_frames_to_mask = int(round(time_percentage * spec.shape[0]))
        t0 = np.random.uniform(low=0.0, high=spec.shape[0] - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = ow
    return spec


class YamNetClassifier:
    def __init__(self, base_path, model_path=None):
        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(self.classes))
        self.feature_name = base_path.split("/")[-2]
        self.feature_number = self.get_feature_number()
        if model_path is not None:
            self.model = load_model(model_path)
        else:
            self.base_path = base_path
            self.do_training()

    def get_feature_number(self):
        if "full" in self.feature_name:
            return 620
        elif "1000" in self.feature_name:
            return 98
        elif "600" in self.feature_name:
            return 58
        elif "300" in self.feature_name:
            return 28
        elif "100" in self.feature_name:
            return 8
        raise Exception("Unexpected feature name.")

    def do_training(self):
        skips = 0
        iters = 1
        bs = 64
        ep = 100
        opts = ["Adagrad", "Adam"]
        lrs = [0.01]
        models = [YAMNet]
        models_name = [x.__name__ for x in models]
        for index, model in enumerate(models):
            for opt in opts:
                for lr in lrs:
                    for iteration in range(iters):
                        if skips > 0:
                            skips -= 1
                            continue

                        print(
                            "\n\n################################################################################\n"
                            "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                            " ###########################\n######################################################" +
                            " ########################\nepochs:", ep, "batch_size:", bs,
                            "\nModel:", models_name[index], "in", models_name,
                            "\nOpt:", opt, "in", opts, "\nLr:", lr, "in", lrs
                        )

                        train_infos = {
                            "iteration": iteration, "batch_size": bs, "epoch": ep,
                            "lr": lr, "opt": opt, "model_name": models_name[index]
                        }

                        m = YAMNet(weights='keras_yamnet/yamnet_conv.h5', classes=7, classifier_activation='softmax',
                                   input_shape=(self.feature_number, 64))
                        self.train_model(train_infos, m)

    def data_gen(self, list_feature_vectors, batch_size, mode="train", aug=None):
        c = 0
        if mode == "train":
            random.shuffle(list_feature_vectors)
        while True:
            labels = []
            features = np.zeros((batch_size, self.feature_number, 64)).astype('float')
            for i in range(c, c + batch_size):
                signal, sound_sr = librosa.load(list_feature_vectors[i], 48000)
                if "full" in self.feature_name and len(signal) < 298368:  # max_length
                    mul = np.tile(signal, 298368 // len(signal))
                    add = signal[:298368 % len(signal)]
                    signal = np.concatenate([mul, add])

                mel = preprocess_input(signal, sound_sr)
                mel = mel if aug is None else aug(mel)
                features[i - c] = np.array(mel)
                labels.append(list_feature_vectors[i].split("/")[-2])
            c += batch_size
            if c + batch_size > len(list_feature_vectors):
                c = 0
                if mode == "train":
                    random.shuffle(list_feature_vectors)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels))
            yield features, labels

    def train_model(self, train_infos, model=None):
        if train_infos["opt"] == "Adam":
            optimizer = Adam(lr=train_infos["lr"])
        elif train_infos["opt"] == "SGD":
            optimizer = SGD(lr=train_infos["lr"])
        else:
            optimizer = Adagrad(lr=train_infos["lr"], decay=1e-6)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_files = glob.glob(self.base_path + "Train/*/*")
        val_files = glob.glob(self.base_path + "Val/*/*")
        train_gen = self.data_gen(train_files, train_infos["batch_size"])
        val_gen = self.data_gen(val_files, train_infos["batch_size"], "val")
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        model_name = "_lr" + str(train_infos["lr"]) + "_Opt" + train_infos["opt"] + "_Model" + \
                     str(train_infos["model_name"]) + "_Feature" + self.feature_name + "_" + \
                     str(train_infos["iteration"]) + ".h5"

        def custom_scheduler(epoch):
            print(0.1 / 10 ** (floor(epoch / 15) + 1))
            return 0.1 / 10 ** (floor(epoch / 15) + 1)

        cb = [ModelCheckpoint(filepath="audio_models/audioModel_{val_accuracy:.4f}_epoch{epoch:02d}" + model_name,
                              monitor="val_accuracy"),
              TensorBoard(log_dir="FULL_AUDIO_LOG", write_graph=True, write_images=True),
              LearningRateScheduler(custom_scheduler)]
        history = model.fit_generator(train_gen, epochs=train_infos["ep"],
                                      steps_per_epoch=(no_of_training_images // train_infos["batch_size"]),
                                      validation_data=val_gen,
                                      validation_steps=(no_of_val_images // train_infos["batch_size"]),
                                      verbose=1, callbacks=cb)

        print("\n\nTrain_Accuracy =", history.history['accuracy'])
        print("\nVal_Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain_Loss =", history.history['loss'])
        print("\nVal_Loss =", history.history['val_loss'])

    def clip_classification(self, path_clip_beginngin):
        all_predictions = {}
        for c in self.classes:
            all_predictions[c] = 0
        val_files = glob.glob(path_clip_beginngin + "*")
        val_gen = self.data_gen(val_files, 1, "eval")
        for batch in val_gen:
            pred = self.lb.inverse_transform(self.model.predict(batch[0]))[0]
            all_predictions[pred] += 1
        return max(all_predictions.items(), key=operator.itemgetter(1))[0]

    def model_eval(self, mode="sample"):
        if mode not in {"sample", "clip"}:
            raise Exception("Evaluation mode not allowed.")
        predictions = []
        ground_truths = []
        stats = []
        if mode == "sample":
            files = glob.glob("/user/vlongobardi/late_feature/" + self.feature_name + "/Val/*/*")
            val_gen = self.data_gen(files, 1, "eval")
            for batch in val_gen:
                ground_truth = self.lb.inverse_transform(batch[1])[0]
                pred = self.lb.inverse_transform(self.model.predict(batch[0]))[0]
                predictions.append(pred)
                ground_truths.append(ground_truth)
        else:
            files = glob.glob("/user/vlongobardi/late_feature/emobase2010_full_wav/Val/*/*")
            files = [file.replace("emobase2010_full_wav", self.feature_name).split(".")[0] for file in files]
            for file in files:
                ground_truths.append(file.split("/")[-2])
                predictions.append(self.clip_classification(file))

        cm = confusion_matrix(ground_truths, predictions, self.classes)
        stats.append(cm)
        stats.append(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2))
        stats.append(accuracy_score(ground_truths, predictions))
        stats.append(classification_report(ground_truths, predictions))

        print("###Results###")
        for index, elem in enumerate(stats):
            if index < 2:
                print_cm(elem, self.classes)
            elif index == 2:
                print("Accuracy score: ", elem)
            else:
                print("Report")
                print(elem)
            print("\n\n")


if __name__ == "__main__":
    audio_path = {
        "e1": "emobase2010_100", "e3": "emobase2010_300",
        "e6": "emobase2010_600", "es": "emobase2010_1000",
        "ef": "emobase2010_full"
    }
    ap = "/user/vlongobardi/late_feature/" + audio_path[sys.argv[1]] + "_wav/"
    print("######################## AUDIO PATH: ", ap)
    ync = YamNetClassifier(base_path=ap)
