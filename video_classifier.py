from tqdm import tqdm
import glob
import sys
import random
import csv
import pickle
import numpy as np
from math import ceil
from os.path import basename, exists, dirname

from keras.models import load_model
from keras.callbacks import ModelCheckpoint  # , TensorBoard
from keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm
from frames_classifier import FramesClassifier
from audio_classifier import AudioClassifier, from_arff_to_feture, get_feature_number
from test_models import *

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class VideoClassifier:

    def __init__(self, train_mode="late_fusion", video_model_path=None, audio_model_path="", time_step=16,
                 base_path="/user/vlongobardi/AFEW/aligned/", feature_name="emobase2010_300"):
        self.time_step = time_step
        self.train_mode = train_mode
        if train_mode == "late_fusion":
            # ac in late serve per generare csv quando necessario e fare la predict per la matrice di confusione
            self.ac = AudioClassifier(audio_model_path)
            self.fc = FramesClassifier(time_step=time_step)
            self.feature_name = '_'.join(audio_model_path.split("Feature")[1].split("_")[0:2])
            print("AC loaded successfully,", audio_model_path, "\nFeature_name:", self.feature_name)
        else:
            self.fc = FramesClassifier(time_step=1)
            self.feature_name = feature_name
        self.feature_num = get_feature_number(self.feature_name)
        self.classes = classes

        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))

        if video_model_path is not None:
            self.model = load_model(video_model_path)
            print("VC loaded successfully", video_model_path)
        else:
            t_files = glob.glob(base_path + "Train" + "/*/*csv")
            v_files = glob.glob(base_path + "Val" + "/*/*csv")
            self.csv_fusion = self.generate_feature(t_files, v_files)
            self.do_training(t_files, v_files)

    def do_training(self, t_files, v_files):
        skips = 0
        iters = 5
        bs = 16
        ep = 50
        opts = ["Adam", "SGD"]
        lrs = [0.1, 0.01, 0.001, 0.0001]
        if self.train_mode == "late_fusion":
            models = [a_model1, a_model2, a_model3, a_model4, a_model5, a_model5_1, a_model5_2, a_model5_3,
                      a_model6, a_model6_1, a_model6_2, a_model7, a_model7_1]
        else:
            models = [early_model_1, early_model_2]
        models_name = [x.__name__ for x in models]
        for index, model in enumerate(models):
            for opt in opts:
                for lr in lrs:
                    for iteration in range(iters):

                        if skips > 0:
                            skips -= 1
                            continue

                        train_infos = {
                            "iteration": iteration, "model_name": models_name[index], "batch_size": bs, "epoch": ep,
                            "lr": lr, "opt": opt
                        }

                        print(
                            "\n\n################################################################################\n"
                            "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                            " ###########################\n######################################################" +
                            " ########################\nepochs:", ep, "batch_size:", bs,
                            "\nmodel:", models_name[index], "in", models_name,
                            "\nopt:", opt, "in", opts,
                            "\nlr:", lr, "in", lrs)

                        # file_name = "videoModel_epoch" + str(ep) + "_lr" + str(lr) + "_Opt" + opt + "_" + \
                        #             models_name[index] + "_Feature" + self.feature_name + "_" + str(
                        #     self.iteration) + "_" + self.train_mode + ".txt"
                        # log_file = open("video_logs/" + file_name, "w")
                        # old_stdout = sys.stdout
                        # sys.stdout = log_file

                        if self.train_mode == "late_fusion":
                            train_infos["generator"] = self.late_gen
                            m = model(14)
                        else:
                            if self.time_step == 1:
                                train_infos["generator"] = self.early_gen
                            else:
                                train_infos["generator"] = self.early_gen_time_step
                            t_files, v_files = self.csv_fusion["train"], self.csv_fusion["val"]
                            m = model(self.feature_num)

                        self.model = self.train(t_files, v_files, train_infos, m)

                        # sys.stdout = old_stdout
                        # log_file.close()

    def generate_feature(self, t_files, v_files):
        if self.train_mode == "late_fusion":
            if not exists('lables_late_fusion' + self.feature_name + '.csv'):
                print("\n##### GENERATING CSV FOR LATE FUSION... #####")
                csv_late_fusion = self._generate_data_for_late_fusion(t_files + v_files)
                print("\n##### CSV GENERATED! #####")
            else:
                csv_late_fusion = {}
                with open('lables_late_fusion' + self.feature_name + '.csv', 'r') as f:
                    f.readline()
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        csv_late_fusion[row[0]] = [row[1], row[2], row[3]]
            return csv_late_fusion
        elif self.train_mode == "early_fusion":
            if len(glob.glob("/user/vlongobardi/early_feature/framefeature/*/*/*.dat")) < 5000:
                print("\n##### GENERATING FEATURES FOR EARLY FUSION... #####")
                self._generate_feature_for_early_fusion(t_files + v_files)
                print("\n##### FEATURES GENERATED! #####")

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
                    with open('features_path_early_fusion_' + name + "_" + self.feature_name + '.csv', 'r') as f:
                        f.readline()
                        csv_reader = csv.reader(f)
                        if self.time_step == 1:
                            for row in csv_reader:
                                csv_early_fusion[name] = [row[0], row[1], row[2], row[3]]
                        else:
                            for clip_id, ground_truth, frame_label, audio_label in csv_reader:
                                if clip_id not in csv_early_fusion:
                                    csv_early_fusion[clip_id] = []
                                    csv_early_fusion[clip_id].append([ground_truth, frame_label, audio_label])
            return csv_early_fusion

    def _generate_data_for_late_fusion(self, total_files):
        my_csv = {}
        total = len(total_files)
        for file in total_files:
            clip_id = file.split(".")[0]
            audio_path = clip_id.replace("AFEW/aligned", self.feature_name)
            label_from_audio = self.ac.clip_classification(audio_path)
            ground_truth, label_from_frame = self.fc.predict(file)
            clip_id = basename(clip_id)
            my_csv[clip_id] = [ground_truth, label_from_frame, label_from_audio]
            print(len(my_csv), "/", total)

        with open('lables_late_fusion' + self.feature_name + '.csv', 'w') as f:
            f.write("clip_id, ground_truth, frame_label, audio_label\n")
            for k in my_csv:
                f.write(str(k) + "," + str(my_csv[k][0]) + "," + str(my_csv[k][1]) + "," + str(my_csv[k][2]) + "\n")
        return my_csv

    def _generate_data_for_early_fusion(self, files, name):
        window_size = int(self.feature_name.split("_")[1])
        frame_to_discard = ceil(window_size / 2 / 40)
        if self.time_step == 1:
            my_csv = []
        else:
            my_csv = {}
        for file in tqdm(files):
            clip_id_temp = file.split(".")[0]
            # '/user/vlongobardi/AFEW/aligned/Train/Angry/012738600.csv'
            # '/user/vlongobardi/early_feature/framefeature/Train/Angry/012738600_0.dat'
            # '/user/vlongobardi/early_feature/emobase2010_600/Train/Angry/012738600_0.arff'
            base_path = clip_id_temp.replace("AFEW/aligned", "early_feature/framefeature") + "_*"
            frames_features_path = glob.glob(base_path)
            audio_features_path = glob.glob(base_path.replace("framefeature", self.feature_name))
            frames_features_path.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            audio_features_path.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ground_truth = basename(dirname(clip_id_temp))
            clip_id = basename(clip_id_temp)

            # discard video frames based on window size
            frames_features_path = frames_features_path[frame_to_discard:]

            for index, audio in enumerate(audio_features_path):
                if self.time_step == 1:
                    my_csv.append([clip_id, ground_truth, frames_features_path[index], audio])

                my_csv[clip_id] = [ground_truth, frames_features_path[index], audio]

        with open('features_path_early_fusion_' + name + "_" + self.feature_name + '.csv', 'w') as f:
            f.write("clip_id, ground_truth, frame_label, audio_label\n")
            if self.time_step == 1:
                for line in my_csv:
                    f.write(line[0] + "," + line[1] + "," + line[2] + "," + line[3] + "\n")
            else:
                for key in my_csv:
                    for line in my_csv[key]:
                        f.write(key + "," + line[0] + "," + line[1] + "," + line[2] + "\n")
        return my_csv

    def _generate_feature_for_early_fusion(self, files):
        """ generate only single frame features (no audio) """
        video_feature_name = "early_feature/framefeature"
        for file_name in tqdm(files):
            features = self.fc.get_feature(file_name)
            base_path = file_name.split(".")[0].replace("AFEW/aligned", video_feature_name)
            for index, feature in enumerate(features):
                with open(base_path + "_" + str(index) + ".dat", 'wb') as f:
                    serialized_feature = pickle.dumps(feature, protocol=0)
                    f.write(serialized_feature)

    def late_gen(self, list_files, batch_size, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_files)
        while True:
            labels = []
            features = np.zeros((batch_size, 2 * len(self.classes))).astype('float')
            for i in range(c, c + batch_size):
                clip_id = basename(list_files[i].split(".")[0])
                ground_truth, label_from_frame, label_from_audio = self.csv_fusion[clip_id]
                features[i - c] = np.append(self.lb.transform(np.array([label_from_audio])),
                                            self.lb.transform(np.array([label_from_frame])))
                labels.append(ground_truth)
            c += batch_size
            if c + batch_size > len(list_files):
                c = 0
                random.shuffle(list_files)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels))
            yield features, labels

    def early_gen(self, list_files, batch_size, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_files)
        while True:
            labels = []
            features = [np.zeros((batch_size, self.fc.time_step, 1024)).astype('float'),  # frame f.
                        np.zeros((batch_size, self.fc.time_step, self.feature_num)).astype('float')]  # audio f.
            for i in range(c, c + batch_size):
                clip_id, ground_truth, frame_feature_path, audio_feature_path = list_files[i]
                with open(frame_feature_path, 'rb') as f:
                    features[0][i - c] = pickle.loads(f.read())
                features[1][i - c] = np.array(from_arff_to_feture(audio_feature_path)).reshape(1, self.feature_num)
                labels.append(ground_truth)
            c += batch_size
            if c + batch_size > len(list_files):
                c = 0
                random.shuffle(list_files)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels)).reshape((16, 1, 7))
            yield features, labels

    def early_gen_time_step(self, list_files, batch_size, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_files)
        while True:
            labels = []
            features = [np.zeros((batch_size, self.time_step, 1024)).astype('float'),  # frame f.
                        np.zeros((batch_size, self.time_step, self.feature_num)).astype('float')]  # audio f.
            for i in range(c, c + batch_size):
                clip_id, ground_truth, frame_feature_path, audio_feature_path = list_files[i]
                with open(frame_feature_path, 'rb') as f:
                    features[0][i - c] = pickle.loads(f.read())
                features[1][i - c] = np.array(from_arff_to_feture(audio_feature_path)).reshape(1, self.feature_num)
                labels.append(ground_truth)
            c += batch_size
            if c + batch_size > len(list_files):
                c = 0
                random.shuffle(list_files)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels)).reshape((16, 7))
            yield features, labels

    def train(self, train_files, val_files, train_data, model):

        if train_data["opt"] == "Adam":
            optimizer = Adam(lr=train_data["lr"])
        else:
            optimizer = SGD(lr=train_data["lr"])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_gen = train_data["generator"](train_files, train_data["batch_size"])
        val_gen = train_data["generator"](val_files, train_data["batch_size"])
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        model_name = "_lr" + str(train_data["lr"]) + "_Opt" + train_data["opt"] + "_Model" + str(
            train_data["model_name"]) + "_Feature" + self.feature_name + "_" + str(
            train_data["iteration"]) + "_" + self.train_mode + ".h5"

        cb = [ModelCheckpoint(filepath=str("video_models/videoModel_{val_accuracy:.4f}_epoch{epoch:02d}" + model_name),
                              monitor="val_accuracy")]
        # cb.append(TensorBoard(log_dir="logs_audio", write_graph=True, write_images=True))
        history = model.fit_generator(train_gen, validation_data=val_gen, epochs=train_data["epochs"],
                                      steps_per_epoch=(no_of_training_images // train_data["batch_size"]),
                                      validation_steps=(no_of_val_images // train_data["batch_size"]),
                                      workers=1, verbose=1, callbacks=cb)
        # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
        print("\n\nTrain Accuracy =", history.history['accuracy'])
        print("\nVal Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain Loss =", history.history['loss'])
        print("\nVal Loss =", history.history['val_loss'])

        model_name = "videoModel_" + str(history.history['val_accuracy'][-1]) + "_epoch" + str(
            train_data["epochs"]) + model_name

        print("\n\nModels saved as:", model_name)
        print("Train:", history.history['accuracy'][-1], "Val:", history.history['val_accuracy'][-1])
        model.save("video_models/" + model_name)

        return model

    def predict(self, path):
        """ ONLY FOR LATE FUSION """
        audio_path = path.split(".")[0].replace("AFEW/aligned", self.feature_name)
        audio_pred = self.ac.clip_classification(audio_path)
        ground_truth, frame_pred = self.fc.predict(path)
        sample = np.append(self.lb.transform(np.array([audio_pred])), self.lb.transform(np.array([frame_pred])))
        # print("\n\n######## 1", sample.shape, sample)
        pred = self.model.predict(sample.reshape((1, 14)))
        return self.lb.inverse_transform(pred)[0], audio_pred, frame_pred, ground_truth

    def print_stats(self, ground_truths, predictions):
        stats = []
        cm = confusion_matrix(ground_truths, predictions, self.classes)
        stats.append(cm)
        stats.append(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2))
        stats.append(accuracy_score(ground_truths, predictions))
        stats.append(classification_report(ground_truths, predictions))

        print("###FULL Results###")
        for index, elem in enumerate(stats):
            if index < 2:
                print_cm(elem, self.classes)
            elif index == 2:
                print("Accuracy score: ", elem)
            else:
                print("Report")
                print(elem)
            print("\n\n")
        print("#################################################################end###\n\n\n")

    def print_confusion_matrix(self, path):
        """ IMPLEMENT FOR EARLY FUSION"""

        labels_late_fusion = {}
        with open('lables_late_fusionemobase2010_600.csv', 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                labels_late_fusion[row[0]] = [row[1], row[2], row[3]]
        predictions = []
        a_p = []
        f_p = []
        ground_truths = []
        files = glob.glob(path + "/*/*csv")
        for file in files:
            # print(file)
            clip_id = basename(file).split(".")[0]
            ground_truth, frame_pred, audio_pred = labels_late_fusion[clip_id]
            sample = np.append(self.lb.transform(np.array([audio_pred])), self.lb.transform(np.array([frame_pred])))
            pred = self.model.predict(sample.reshape((1, 14)))
            pred = self.lb.inverse_transform(pred)[0]
            # print(ground_truth, frame_pred, audio_pred, pred)
            # pred, audio_pred, frame_pred, ground_truth = self.predict(file)
            predictions.append(pred)
            a_p.append(audio_pred)
            f_p.append(frame_pred)
            ground_truths.append(ground_truth)

        self.print_stats(ground_truths, predictions)
        self.print_stats(ground_truths, a_p)
        self.print_stats(ground_truths, f_p)


if __name__ == "__main__":
    if sys.argv[1] == "late":
        print("LATE")
        # "audio_models/audioModel_0.2953_epoch14_lr0.001_OptAdam_Modela_model6_1_Featureemobase2010_600_0.h5"
        model_path = "audio_models/audioModel_0.3668_epoch37_lr0.001_OptSGD_Modela_model7_Featureemobase2010_full_1.h5"
        vc = VideoClassifier(train_mode="late_fusion", audio_model_path=model_path)
        # vc.print_confusion_matrix("/user/vlongobardi/AFEW/aligned/Val")
    else:
        if "t" in sys.argv[1]:
            ts = 16
        else:
            ts = 1
        print("EARLY")
        arff_paths = {"e1": "emobase2010_100", "i1": "IS09_emotion_100",
                      "e3": "emobase2010_300", "i3": "IS09_emotion_300",
                      "e6": "emobase2010_600", "i6": "IS09_emotion_600"}

        arff_path = arff_paths[sys.argv[1].replace("t", "")]
        vc = VideoClassifier(train_mode="early_fusion", time_step=ts, feature_name=arff_path)
