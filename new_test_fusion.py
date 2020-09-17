import csv
import glob
import random
import sys
from math import ceil, floor
from os.path import basename, exists, dirname, isfile

import numpy as np
import keras
from keras import Model, Input, regularizers
from keras.layers import TimeDistributed, LSTMCell, Reshape, Dense, Lambda, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, accuracy_score  # , classification_report
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from Dataset.Dataset_Utils.augmenter import NoAug
from Dataset.Dataset_Utils.datagen import DataGenerator as DataGen
from Dataset.Dataset_Utils.dataset_tools import print_cm
from Models.model_sharma import SharmaNet
from audio_classifier import AudioClassifier, from_arff_to_feture
from frames_classifier import FramesClassifier
from test_models import *

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def my_model():
    r1, r2 = regularizers.l2(1e-5), regularizers.l2(1e-5)
    frame_input = Input(shape=(16, 1024))
    audio_input = Input(shape=(16, 1582))
    x = Concatenate(name='fusion1')([frame_input, audio_input])
    x = TimeDistributed(Dense(100, activation='tanh', kernel_regularizer=r1, name='ff_logit_lstm'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(7, activation='softmax', kernel_regularizer=r2, name='ff_logit'))(x)
    x = Lambda(lambda y: tf.reduce_mean(y, axis=1))(x)
    return Model([audio_input, frame_input], x)


class VideoClassifier:

    def __init__(self, train_mode="late_fusion", video_model_path=None, time_step=16,
                 base_path="/user/vlongobardi/AFEW/aligned/", feature_name="emobase2010_100", stride=1):
        self.time_step = time_step
        self.train_mode = train_mode
        self.feature_name = feature_name
        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))
        self.feature_num = 1582
        self.offset = ceil(int(self.feature_name.split("_")[1]) / 2 / 40)
        self.stride = stride

        if video_model_path is not None:
            self.model = my_model()
            self.model.load_weights(video_model_path)
            print("VideoClassifier loaded successfully", video_model_path)
        else:
            t_files = glob.glob(base_path + "Train" + "/*/*csv")
            v_files = glob.glob(base_path + "Val" + "/*/*csv")
            self.csv_fusion = self.generate_feature(t_files, v_files)
            self.do_training()

    def do_training(self):
        skips = 0
        iters = 1
        bs = 16
        ep = 75
        opts = ["SGD", "Adam"]
        lrs = [0.01]
        models = [my_model]
        models_name = [x.__name__ for x in models]
        for index, model in enumerate(models):
            for opt in opts:
                for lr in lrs:
                    for iteration in range(iters):

                        if skips > 0:
                            skips -= 1
                            continue

                        train_infos = {
                            "iteration": iteration, "model_name": models_name[index],
                            "batch_size": bs, "epoch": ep, "lr": lr, "opt": opt
                        }

                        print(
                            "\n\n################################################################################\n"
                            "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                            " ###########################\n######################################################" +
                            " ########################\nepochs:", ep, "batch_size:", bs, "\nmodel:", models_name[index],
                            "in", models_name, "\nopt:", opt, "in", opts, "\nlr:", lr, "in", lrs)

                        train_infos["generator1"] = self.early_gen_train
                        train_infos["generator2"] = self.early_gen_new_val
                        t_files, v_files = self.csv_fusion["train"], self.csv_fusion["val"]
                        m = model()

                        self.train(t_files, v_files, train_infos, m)

    def generate_feature(self, t_files, v_files):
        if not exists('features_path_early_fusion_train_' + self.feature_name + '.csv'):
            print("\n##### GENERATING CSV FOR EARLY FUSION... #####")
            csv_early_fusion = {
                "train": self._generate_data_for_early_fusion(t_files, "train"),
                "val": self._generate_data_for_early_fusion(v_files, "val")
            }
            print("\n##### CSV GENERATED! #####")
        else:
            csv_early_fusion = {}
            for name in ["train", "val"]:
                csv_early_fusion[name] = self.load_early_csv(name)
        return csv_early_fusion

    def load_early_csv(self, dataset):
        csv_early_fusion = {}
        print("Opening csv: features_path_early_fusion_" + dataset + "_" + self.feature_name + '.csv')
        with open('features_path_early_fusion_' + dataset + "_" + self.feature_name + '.csv', 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for clip_id, ground_truth, frame_label, audio_label in csv_reader:
                if clip_id not in csv_early_fusion:
                    csv_early_fusion[clip_id] = []
                csv_early_fusion[clip_id].append([ground_truth, frame_label, audio_label])
        return csv_early_fusion

    def _generate_data_for_early_fusion(self, files, name):
        # '/user/vlongobardi/AFEW/aligned/Train/Angry/012738600.csv'
        # '/user/vlongobardi/early_feature/framefeature/Train/Angry/012738600_0.dat'
        # '/user/vlongobardi/early_feature/emobase2010_600/Train/Angry/012738600_0.arff'
        if "full" in self.feature_name:
            frame_to_discard = 0
        else:
            window_size = int(self.feature_name.split("_")[1])
            frame_to_discard = ceil(window_size / 2 / 40)
        my_csv = {}
        for file in tqdm(files):
            clip_id_temp = file.split(".")[0]
            base_path = clip_id_temp.replace("AFEW/aligned", "early_feature/framefeature") + "*"
            frames_features_path = glob.glob(base_path)
            audio_features_path = glob.glob(
                base_path.replace("early_feature/framefeature", "early_feature/" + self.feature_name))
            frames_features_path.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if "full" not in self.feature_name:
                audio_features_path.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ground_truth = basename(dirname(clip_id_temp))
            clip_id = basename(clip_id_temp)

            # discard video frames based on window size
            frames_features_path = frames_features_path[frame_to_discard:]
            if len(frames_features_path) < 16:
                continue
                # print("FRAME TOO FEW SAMPLES:", len(frames_features_path), clip_id)
            if len(audio_features_path) < 16 and "full" not in self.feature_name:
                continue
                # print("AUDIO TOO FEW SAMPLES:", len(audio_features_path), clip_id)
            for index, frame in enumerate(frames_features_path):
                if clip_id not in my_csv.keys():
                    my_csv[clip_id] = []
                if "full" not in self.feature_name:
                    my_csv[clip_id].append([ground_truth, frame, audio_features_path[index]])
                else:
                    my_csv[clip_id].append([ground_truth, frame, audio_features_path[0]])
        with open('features_path_early_fusion_' + name + "_" + self.feature_name + '.csv', 'w') as f:
            f.write("clip_id, ground_truth, frame_label, audio_label\n")
            for key in my_csv:
                for line in my_csv[key]:
                    f.write(key + "," + line[0] + "," + line[1] + "," + line[2] + "\n")
        return my_csv

    def early_gen_train(self, list_files, batch_size):
        c = 0
        clip_ids = list(self.csv_fusion["train"].keys())
        random.shuffle(clip_ids)
        while True:
            labels = []
            features = [np.zeros((batch_size, self.time_step, self.feature_num)).astype('float'),
                        np.zeros((batch_size, self.time_step, 1024)).astype('float')]
            for i in range(c, c + batch_size):
                clip_id = clip_ids[i]
                video_info = self.csv_fusion["train"][clip_id]
                ground_truth = video_info[0][0]

                # first_frame_num = int(video_info[0][1].split("_")[-1].split(".")[0])
                start = random.randint(0, len(video_info) - self.time_step)
                for index, elem in enumerate(video_info[start:self.time_step + start]):
                    _, frame_path, audio_path = elem
                    if not isfile(frame_path):
                        start += 1
                        if start >= len(video_info):
                            raise
                        continue
                    frame_feature = np.load(frame_path)
                    features[0][i - c][index] = np.array(from_arff_to_feture(audio_path)).reshape(self.feature_num, )
                    features[1][i - c][index] = frame_feature.reshape(1024,)
                labels.append(ground_truth)
            c += batch_size
            if c + batch_size > len(clip_ids):
                c = 0
            random.shuffle(clip_ids)
            labels = self.lb.transform(np.array(labels)).reshape((batch_size, 7))
            yield features, labels

    def early_gen_new_val(self, list_files, batch_size, mode="val", stride=1):
        """ stride 50% sul su tutti i file """
        c = 0
        labels = features = []
        clip_ids = list(list_files.keys())
        while True:
            for clip_id in clip_ids:
                video_info = list_files[clip_id]
                ground_truth = video_info[0][0]

                for start in range(0, len(video_info) - self.time_step, self.time_step // stride):
                    if c == 0:
                        labels = []
                        features = [np.zeros((batch_size, self.time_step, self.feature_num)).astype('float'),
                                    np.zeros((batch_size, self.time_step, 1024)).astype('float')]
                    for index, elem in enumerate(video_info[start:self.time_step + start]):
                        _, frame_path, audio_path = elem
                        frame_feature = np.load(frame_path)
                        features[0][c][index] = np.array(from_arff_to_feture(audio_path)).reshape(
                            self.feature_num, )
                        features[1][c][index] = frame_feature.reshape(1024,)
                    labels.append(ground_truth)

                    c += 1
                    if c == batch_size:
                        c = 0
                        labels = self.lb.transform(np.array(labels)).reshape((batch_size, 7))
                        yield features, labels
            if mode == "eval":
                break

    def early_gen_test_clip(self, list_files, clip_id, stride=1):
        """ stride su singolo file, quindi va richiamato per ogni file """
        ground_truth = list_files[0][0]
        csv_path = '/user/vlongobardi/AFEW/aligned/Val/GroundTruth/ID.csv'
        csv_path = csv_path.replace("GroundTruth", ground_truth).replace("ID", clip_id)
        first_frame_num = int(list_files[0][1].split("_")[-1].split(".")[0])
        start = 0
        end = len(list_files) - self.time_step
        while True:
            labels = []
            features = [np.zeros((1, self.time_step, self.feature_num)).astype('float'),
                        np.zeros((1, self.time_step, 1024)).astype('float')]
            for index, elem in enumerate(list_files[start:start + self.time_step]):
                _, frame_path, audio_path = elem
                frame_feature = np.load(frame_path)
                features[0][0][index] = np.array(from_arff_to_feture(audio_path)).reshape(self.feature_num, )
                features[1][0][index] = frame_feature.reshape(1024,)
            labels.append(ground_truth)
            start += self.time_step // stride
            if start >= end:
                break
            labels = self.lb.transform(np.array(labels)).reshape((1, 7))
            yield features, labels

    def get_validation_dim(self):
        if self.stride == 2:
            if "full" in self.feature_name:
                return 141
            elif "600" in self.feature_name:
                return 0
            elif "300" in self.feature_name:
                return 114
            elif "100" in self.feature_name:
                return 128
        elif self.stride == 1:
            if "full" in self.feature_name:
                return 76
            elif "600" in self.feature_name:
                return 0
            elif "300" in self.feature_name:
                return 63
            elif "100" in self.feature_name:
                return 69
        elif self.stride == self.time_step:
            return 0

    def train(self, train_files, val_files, train_data, model):
        if train_data["opt"] == "Adam":
            optimizer = Adam(lr=train_data["lr"])
        else:
            optimizer = SGD(lr=train_data["lr"])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_gen = train_data["generator1"](train_files, train_data["batch_size"])
        no_of_training_images = len(train_files)

        if self.train_mode == "early_fusion":
            no_of_val_images = self.get_validation_dim()
            # val_gen = train_data["generator2"](val_files, train_data["batch_size"], stride)
        else:
            no_of_val_images = len(val_files)
            # val_gen = train_data["generator2"](val_files, train_data["batch_size"])

        #  stride = 1,             no overlapping
        #  stride = 2,             overlapping: 50%
        #  stride = time_step,     stride: 1

        model_name = "_lr" + str(train_data["lr"]) + "_Opt" + train_data["opt"] + "_Model" + str(
            train_data["model_name"]) + "_Feature" + self.feature_name + "_" + str(
            train_data["iteration"]) + "_" + self.train_mode  # + "_modelType" + str(self.model_type)
        if self.train_mode == "early_fusion":
            model_name += "stride" + str(self.stride)
        model_name += ".h5"

        def custom_scheduler(epoch):
            print(0.1 / 10 ** (floor(epoch / 25) + 1))
            return 0.1 / 10 ** (floor(epoch / 25) + 1)

        class CheckValCMCallback(keras.callbacks.Callback):
            def __init__(self, m, dim, validation_files, epoch):
                super().__init__()
                self.vc = m
                self.dim = dim
                self.val_files = validation_files
                self.epoch = epoch
                self.accs = []

            def on_epoch_end(self, epoch, logs=None):
                if self.vc.train_mode == "early_fusion":
                    csv_fusion = self.vc.load_early_csv("val")
                    # gen = self.vc.early_gen_new_val(csv_fusion, 16, "eval")
                    # predictions = []
                    # ground_truths = []
                    # for x in gen:
                    #     ground_truths.append(self.vc.lb.inverse_transform(x[1])[0])
                    #     pred = self.model.predict(x[0])
                    #     pred = self.vc.lb.inverse_transform(pred)
                    #     predictions.append(pred[0])
                    #     self.vc.print_stats(ground_truths, predictions, "Video" + str(epoch))
                    gen = self.vc.early_gen_new_val(csv_fusion, 16, "eval")
                else:
                    gen = self.vc.late_gen(self.val_files, 16, "eval")
                acc = self.model.evaluate_generator(gen, self.dim, workers=0)
                self.accs.append(acc)
                print("Evaluate:", acc)

                if self.epoch == epoch + 1:
                    print("Validation_Accuracy =", self.accs)

        cb = [ModelCheckpoint(
            filepath=str(
                "weights_new_fusion/videoModel__t{accuracy:.4f}_epoch{epoch:02d}" + model_name),
            monitor="val_accuracy", save_weights_only=True),
            TensorBoard(log_dir="NewFusionLogs/" + self.train_mode + "/" + self.feature_name, write_graph=True,
                        write_images=True)]
        if self.train_mode == "early_fusion":
            cb += [LearningRateScheduler(custom_scheduler)]
        cb += [CheckValCMCallback(self, no_of_val_images, val_files, train_data["epoch"])]
        history = model.fit_generator(train_gen,
                                      # validation_data=val_gen,
                                      epochs=train_data["epoch"],
                                      steps_per_epoch=(no_of_training_images * 2 // train_data["batch_size"]),
                                      # validation_steps=(no_of_val_images // train_data["batch_size"]),
                                      workers=0, verbose=1, callbacks=cb)
        print("\n\nTrain_Accuracy =", history.history['accuracy'])
        # print("\nVal_Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain_Loss =", history.history['loss'])
        # print("\nVal_Loss =", history.history['val_loss'])

    def print_stats(self, ground_truths, predictions, name):
        cm = confusion_matrix(ground_truths, predictions, self.classes)
        print("###" + name + " Results###\n")
        # print_cm(cm, self.classes)
        # print("\n\n")
        print_cm(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3), self.classes)
        print("\n\n")
        print("Accuracy score: ", accuracy_score(ground_truths, predictions), "\n\n")
        # print("Report")
        # print(classification_report(ground_truths, predictions))
        print("#################################################################end###\n\n\n")

    def print_confusion_matrix(self, stride=1):
        """ IMPLEMENT FOR EARLY FUSION MISSING """
        csv_fusion = {}
        predictions = []
        ground_truths = []
        if self.train_mode == "early_fusion":
            csv_fusion = self.load_early_csv("val")
            print("CSV loaded", len(csv_fusion))
            gen = self.early_gen_new_val(csv_fusion, 1, "eval", stride)
            for x in gen:
                ground_truths.append(self.lb.inverse_transform(x[1])[0])
                pred = self.model.predict(x[0])
                pred = self.lb.inverse_transform(pred)
                predictions.append(pred[0])
                # print("\ngt, pred", self.lb.inverse_transform(x[1]), pred)
            self.print_stats(ground_truths, predictions, "Video")
        else:
            with open('lables_late_fusion' + self.feature_name + '.csv', 'r') as f:
                f.readline()
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    csv_fusion[row[0]] = [row[1], row[2], row[3]]
            a_p = []
            f_p = []
            files = glob.glob("/user/vlongobardi/late_feature/" + self.feature_name + "/*/*csv")
            for file in files:
                clip_id = basename(file).split(".")[0]
                ground_truth, frame_pred, audio_pred = csv_fusion[clip_id]
                sample = np.append(self.lb.transform(np.array([audio_pred])), self.lb.transform(np.array([frame_pred])))
                pred = self.model.predict(sample.reshape((1, 14)))
                pred = self.lb.inverse_transform(pred)[0]
                predictions.append(pred)
                a_p.append(audio_pred)
                f_p.append(frame_pred)
                ground_truths.append(ground_truth)

            self.print_stats(ground_truths, predictions, "Video")
            self.print_stats(ground_truths, a_p, "Audio")
            self.print_stats(ground_truths, f_p, "Frame")


if __name__ == "__main__":
    if sys.argv[1] == "late":
        print("LATE")
        model_path = [
            "audio_models/audioModel_0.2285_epoch135_lr0.1_OptSGD_Modela_model7_Featureemobase2010_100_3.h5",
            "audio_models/audioModel_0.2650_epoch01_lr0.01_OptSGD_Modela_model7_Featureemobase2010_300_2.h5",
            "audio_models/audioModel_0.2865_epoch13_lr0.001_OptSGD_Modela_model7_Featureemobase2010_600_0.h5",
            "audio_models/audioModel_0.3668_epoch67_lr0.001_OptSGD_Modela_model7_Featureemobase2010_full_2.h5"
        ]
        for mp in model_path:
            vc = VideoClassifier(train_mode="late_fusion", audio_model_path=mp)
    elif sys.argv[1] == "early":
        # mt = int(sys.argv[2])
        print("EARLY")  # , Model_type:", mt)
        arff_paths = {"e1": "emobase2010_100", "i1": "IS09_emotion_100",
                      "e3": "emobase2010_300", "i3": "IS09_emotion_300",
                      "e6": "emobase2010_600", "i6": "IS09_emotion_600",
                      "ef": "emobase2010_full", "if": "IS09_emotion_full"}
        vc = VideoClassifier(train_mode="early_fusion", feature_name=arff_paths[sys.argv[2]])  # , model_type=mt)
