import csv
import traceback
import glob
import random
import sys
from math import ceil, floor
from os.path import basename, exists, dirname

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from Dataset.Dataset_Utils.augmenter import NoAug
from Dataset.Dataset_Utils.datagen import DataGenerator as DataGen
from Dataset.Dataset_Utils.dataset_tools import print_cm
from Models.model_sharma import SharmaNet
from audio_classifier import AudioClassifier, from_arff_to_feture, get_feature_number
from frames_classifier import FramesClassifier
from test_models import *

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class VideoClassifier:

    def __init__(self, train_mode="late_fusion", video_model_path=None, audio_model_path="", time_step=16,
                 base_path="/user/vlongobardi/AFEW/aligned/", feature_name="emobase2010_300", model_type=0, stride=2):
        self.time_step = time_step
        self.train_mode = train_mode
        if train_mode == "late_fusion":
            # ac in late serve per generare csv quando necessario e fare la predict per la matrice di confusione
            self.ac = AudioClassifier(audio_model_path)
            self.fc = FramesClassifier(time_step=time_step)
            self.feature_name = '_'.join(audio_model_path.split("Feature")[1].split("_")[0:2])
            print("AC loaded successfully,", audio_model_path, "\nFeature_name:", self.feature_name)
        else:
            self.feature_name = feature_name
            self.model_type = model_type
            self.stride = stride
        self.feature_num = get_feature_number(self.feature_name)
        self.classes = classes

        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))

        if video_model_path is not None:
            if train_mode == "late_fusion":
                self.model = load_model(video_model_path)
                print("VideoClassifier loaded successfully", video_model_path)
            else:
                self.model = SharmaNet((self.time_step, 224, 224, 3), dim=self.model_type)
                self.model.load_weights(video_model_path)
                print("VideoClassifier loaded successfully", video_model_path)
        else:
            t_files = glob.glob(base_path + "Train" + "/*/*csv")
            v_files = glob.glob(base_path + "Val" + "/*/*csv")
            self.csv_fusion = self.generate_feature(t_files, v_files)
            self.do_training(t_files, v_files)

    def do_training(self, t_files, v_files):
        skips = 0
        iters = 1
        bs = 16
        ep = 150
        opts = ["SGD", "Adam"]
        #lrs = [0.1, 0.01, 0.001, 0.0001]
        lrs = [0.01]
        if self.train_mode == "late_fusion":
            models = [a_model5_2, a_model5_3, a_model7, a_model7_1]
        else:
            models = [SharmaNet]
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
                            train_infos["generator1"] = self.early_gen_train
                            train_infos["generator2"] = self.early_gen_new_val
                            t_files, v_files = self.csv_fusion["train"], self.csv_fusion["val"]
                            m = model((self.time_step, 224, 224, 3), dim=self.model_type)

                        self.model = self.train(t_files, v_files, train_infos, m)
                        del self.model
                        # sys.stdout = old_stdout
                        # log_file.close()

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
        if "full" in self.feature_name:
            frame_to_discard = 0
        else:
            window_size = int(self.feature_name.split("_")[1])
            frame_to_discard = ceil(window_size / 2 / 40)
        my_csv = {}
        for file in tqdm(files):
            clip_id_temp = file.split(".")[0]
            # '/user/vlongobardi/AFEW/aligned/Train/Angry/012738600.csv'
            # '/user/vlongobardi/early_feature/framefeature/Train/Angry/012738600_0.dat'
            # '/user/vlongobardi/early_feature/emobase2010_600/Train/Angry/012738600_0.arff'
            base_path = clip_id_temp.replace("AFEW/aligned", "early_feature/framefeature") + "*"
            frames_features_path = glob.glob(base_path)
            audio_features_path = glob.glob(
                base_path.replace("early_feature/framefeature", "late_feature/" + self.feature_name))
            frames_features_path.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if "full" not in self.feature_name:
                audio_features_path.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ground_truth = basename(dirname(clip_id_temp))
            clip_id = basename(clip_id_temp)

            # discard video frames based on window size
            frames_features_path = frames_features_path[frame_to_discard:]
            if len(frames_features_path) < 16:
                continue
                print("FRAME TOO FEW SAMPLES:", len(frames_features_path), clip_id)
            if len(audio_features_path) < 16 and "full" not in self.feature_name:
                continue
                print("AUDIO TOO FEW SAMPLES:", len(audio_features_path), clip_id)
            for index, frame in enumerate(frames_features_path):
                try:
                    if clip_id not in my_csv.keys():
                        my_csv[clip_id] = []
                    if "full" not in self.feature_name:
                        my_csv[clip_id].append([ground_truth, frame, audio_features_path[index]])
                    else:
                        my_csv[clip_id].append([ground_truth, frame, audio_features_path[0]])
                except:
                    print(".")
                    continue
        with open('features_path_early_fusion_' + name + "_" + self.feature_name + '.csv', 'w') as f:
            f.write("clip_id, ground_truth, frame_label, audio_label\n")
            for key in my_csv:
                for line in my_csv[key]:
                    f.write(key + "," + line[0] + "," + line[1] + "," + line[2] + "\n")
        return my_csv

    def late_gen(self, list_files, batch_size, mode="Train"):
        c = 0
        if mode == "Train":
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
                if mode == "Train":
                    random.shuffle(list_files)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels))
            yield features, labels

    def early_gen_train(self, list_files, batch_size):
        c = 0
        clip_ids = list(list_files.keys())
        random.shuffle(clip_ids)
        while True:
            labels = []
            features = [np.zeros((batch_size, self.feature_num)).astype('float')] * self.time_step
            features.append(np.zeros((batch_size, self.time_step, 224, 224, 3)).astype('float'))
            for i in range(c, c + batch_size):
                clip_id = clip_ids[i]
                video_info = list_files[clip_id]
                ground_truth = video_info[0][0]
                csv_path = '/user/vlongobardi/AFEW/aligned/Train/GroundTruth/ID.csv'
                csv_path = csv_path.replace("GroundTruth", ground_truth).replace("ID", clip_id)
                images = DataGen(csv_path, '', 1, 31, NoAug(), 16, 1, 12, test=True)[0][0][0]
                first_frame_num = int(video_info[0][1].split("_")[-1].split(".")[0])
                start = random.randint(0, len(video_info) - self.time_step)
                for index, elem in enumerate(video_info[start:self.time_step + start]):
                    ground_truth, _, audio_path = elem
                    features[-1][i - c][index] = images[first_frame_num + start + index]
                    features[index][i - c] = np.array(from_arff_to_feture(audio_path)).reshape(self.feature_num, )
                labels.append(ground_truth)
            c += batch_size
            if c + batch_size > len(clip_ids):
                c = 0
                random.shuffle(clip_ids)
            labels = self.lb.transform(np.array(labels)).reshape((batch_size, 7))
            yield features, labels

    def early_gen_old_val(self, list_files, batch_size, mode):
        c = 0
        clip_ids = list(list_files.keys())
        random.shuffle(clip_ids)
        while True:
            labels = []
            features = [np.zeros((batch_size, self.feature_num)).astype('float')] * self.time_step
            features.append(np.zeros((batch_size, self.time_step, 224, 224, 3)).astype('float'))
            for i in range(c, c + batch_size):
                clip_id = clip_ids[i]
                video_info = list_files[clip_id]
                ground_truth = video_info[0][0]
                csv_path = '/user/vlongobardi/AFEW/aligned/Val/GroundTruth/ID.csv'
                csv_path = csv_path.replace("GroundTruth", ground_truth).replace("ID", clip_id)
                images = DataGen(csv_path, '', 1, 31, NoAug(), 16, 1, 12, test=True)[0][0][0]
                first_frame_num = int(video_info[0][1].split("_")[-1].split(".")[0])
                start = random.randint(0, len(video_info) - self.time_step)
                for index, elem in enumerate(video_info[start:self.time_step + start]):
                    ground_truth, _, audio_path = elem
                    features[-1][i - c][index] = images[first_frame_num + start + index]
                    features[index][i - c] = np.array(from_arff_to_feture(audio_path)).reshape(self.feature_num, )
                labels.append(ground_truth)
            c += batch_size
            if c + batch_size > len(clip_ids):
                c = 0
                if mode == "eval":
                    break
                random.shuffle(clip_ids)
            labels = self.lb.transform(np.array(labels)).reshape((batch_size, 7))
            yield features, labels

    def early_gen_new_val(self, list_files, batch_size, mode="val", stride=2):
        """ stride 50% sul su tutti i file """
        c = 0
        clip_ids = list(list_files.keys())
        while True:
            try:
                for clip_id in clip_ids:
                    video_info = list_files[clip_id]
                    ground_truth = video_info[0][0]
                    csv_path = '/user/vlongobardi/AFEW/aligned/Val/GroundTruth/ID.csv'
                    csv_path = csv_path.replace("GroundTruth", ground_truth).replace("ID", clip_id)
                    images = DataGen(csv_path, '', 1, 31, NoAug(), 16, 1, 12, test=True)[0][0][0]
                    first_frame_num = int(video_info[0][1].split("_")[-1].split(".")[0])

                    for start in range(0, len(video_info) - self.time_step, self.time_step // stride):
                        if c == 0:
                            labels = []
                            features = [np.zeros((batch_size, self.feature_num)).astype('float')] * self.time_step
                            features.append(np.zeros((batch_size, self.time_step, 224, 224, 3)).astype('float'))

                        for index, elem in enumerate(video_info[start:self.time_step + start]):
                            audio_path = elem[2]
                            features[-1][c][index] = images[first_frame_num + start + index]
                            features[index][c] = np.array(from_arff_to_feture(audio_path)).reshape(self.feature_num, )
                        labels.append(ground_truth)

                        c += 1
                        if c == batch_size:
                            c = 0
                            labels = self.lb.transform(np.array(labels)).reshape((batch_size, 7))
                            yield features, labels
            except Exception as ex:
                print("\n\nEXCEPTION")
                traceback.print_exception(type(ex), ex, ex.__traceback__)
                print("\nclip_index:", clip_id, "\nlen(clip_ids)", len(clip_ids))
                print("\ncsv_path", csv_path, "\nstart", start, "\nc", c)
            if mode == "eval":
                break

    def early_gen_test_clip(self, list_files, clip_id, stride=2):
        """ stride su singolo file, quindi va richiamato per ogni file """
        ground_truth = list_files[0][0]
        csv_path = '/user/vlongobardi/AFEW/aligned/Val/GroundTruth/ID.csv'
        csv_path = csv_path.replace("GroundTruth", ground_truth).replace("ID", clip_id)
        first_frame_num = int(list_files[0][1].split("_")[-1].split(".")[0])
        start = 0
        end = len(list_files) - self.time_step
        while True:
            labels = []
            features = [np.zeros((1, self.feature_num)).astype('float')] * self.time_step
            features.append(np.zeros((1, self.time_step, 224, 224, 3)).astype('float'))
            images = DataGen(csv_path, '', 1, 31, NoAug(), 16, 1, 12, test=True)[0][0][0]
            for index, elem in enumerate(list_files[start:start + self.time_step]):
                audio_path = elem[2]
                features[-1][0][index] = images[first_frame_num + start + index]
                features[index][0] = np.array(from_arff_to_feture(audio_path)).reshape(self.feature_num, )
            labels.append(ground_truth)
            start += self.time_step // stride
            if start >= end:
                break
            labels = self.lb.transform(np.array(labels)).reshape((1, 7))
            yield features, labels

    def train(self, train_files, val_files, train_data, model):
        if train_data["opt"] == "Adam":
            optimizer = Adam(lr=train_data["lr"])
        else:
            optimizer = SGD(lr=train_data["lr"])

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        stride = stride = self.stride

        train_gen = train_data["generator1"](train_files, train_data["batch_size"])
        val_gen = train_data["generator2"](val_files, train_data["batch_size"], stride)
        no_of_training_images = len(train_files)

        if stride == 2:
            no_of_val_images = 1461
        elif stride == 1:
            no_of_val_images = 811
        elif stride == self.time_step:
            no_of_val_images = 10518
        else:
            no_of_val_images = len(val_files)

        # no_of_val_images = 50   # stride = 1,             no overlapping
        # no_of_val_images = 91   # stride = 2,             stride/overlapping: 50%
        # no_of_val_images = 657  # stride = time_step,     stride: 1

        model_name = "_lr" + str(train_data["lr"]) + "_Opt" + train_data["opt"] + "_Model" + str(
            train_data["model_name"]) + "_Feature" + self.feature_name + "_" + str(
            train_data["iteration"]) + "_" + self.train_mode + "_modelType" + str(self.model_type) + ".h5"

        def custom_scheduler(epoch):
            print(0.1 / 10 ** (floor(epoch / 30) + 1))
            return 0.1 / 10 ** (floor(epoch / 30) + 1)

        cb = [ModelCheckpoint(
            filepath=str(
                "ultimate_early_weights/videoModel_v{val_accuracy:.4f}_t{accuracy:.4f}_epoch{epoch:02d}" + model_name),
            monitor="val_accuracy", save_weights_only=True),
            TensorBoard(log_dir="Ultimate_logs_" + self.train_mode + str(self.model_type), write_graph=True, write_images=True)]
        if self.train_mode == "early_fusion":
            cb += [LearningRateScheduler(custom_scheduler)]
        history = model.fit_generator(train_gen, validation_data=val_gen, epochs=train_data["epoch"],
                                      steps_per_epoch=(no_of_training_images // train_data["batch_size"]),
                                      validation_steps=(no_of_val_images // train_data["batch_size"]),
                                      workers=1, verbose=1, callbacks=cb)
        print("\n\nTrain_Accuracy =", history.history['accuracy'])
        print("\nVal_Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain_Loss =", history.history['loss'])
        print("\nVal_Loss =", history.history['val_loss'])

        model_name = "videoModel_" + str(history.history['val_accuracy'][-1]) + "_epoch" + str(
            train_data["epoch"]) + model_name

        print("\n\nModels saved as:", model_name)
        print("Train:", history.history['accuracy'][-1], "Val:", history.history['val_accuracy'][-1])
        # model.save("video_models/" + model_name)
        model.save_weights("video_models_early_weights/" + model_name)

        return model

    def print_stats(self, ground_truths, predictions, name):
        cm = confusion_matrix(ground_truths, predictions, self.classes)
        print("###" + name + " Results###\n")
        # print_cm(cm, self.classes)
        # print("\n\n")
        print_cm(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3), self.classes)
        print("\n\n")
        print("Accuracy score: ", accuracy_score(ground_truths, predictions), "\n\n")
        print("Report")
        print(classification_report(ground_truths, predictions))
        print("#################################################################end###\n\n\n")

    def print_confusion_matrix(self, path, stride):
        """ IMPLEMENT FOR EARLY FUSION MISSING """
        csv_fusion = {}
        predictions = []
        ground_truths = []
        if self.train_mode == "early_fusion":
            csv_fusion = self.load_early_csv("val")
            print("CSV loaded", len(csv_fusion))
            gen = self.early_gen_new_val(csv_fusion, 1, "eval", stride)
            for x in gen:
                ground_truths.append(vc.lb.inverse_transform(x[1])[0])
                pred = vc.model.predict(x[0])
                pred = vc.lb.inverse_transform(pred)
                predictions.append(pred[0])
                # print("\ngt, pred", self.lb.inverse_transform(x[1]), pred)
            vc.print_stats(ground_truths, predictions, "Video")
        else:
            with open('lables_late_fusion' + self.feature_name + '.csv', 'r') as f:
                f.readline()
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    csv_fusion[row[0]] = [row[1], row[2], row[3]]
            a_p = []
            f_p = []
            files = glob.glob(path + "/*/*csv")
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
        # "audio_models/audioModel_0.2953_epoch14_lr0.001_OptAdam_Modela_model6_1_Featureemobase2010_600_0.h5"
        model_path = "audio_models/audioModel_0.3668_epoch37_lr0.001_OptSGD_Modela_model7_Featureemobase2010_full_1.h5"
        model_path = "audio_models/audioModel_0.3696_epoch60_lr0.001_OptSGD_Modela_model7_Featureemobase2010_full_2.h5"
        vc = VideoClassifier(train_mode="late_fusion", audio_model_path=model_path)
        # vc.print_confusion_matrix("/user/vlongobardi/AFEW/aligned/Val")
    else:
        mts = [int(sys.argv[2])]
        print("EARLY")
        for mt in mts:
            print("Model_type:", mt)
            arff_paths = {"e1": "emobase2010_100", "i1": "IS09_emotion_100",
                          "e3": "emobase2010_300", "i3": "IS09_emotion_300",
                          "e6": "emobase2010_600", "i6": "IS09_emotion_600",
                          "ef": "emobase2010_full", "if": "IS09_emotion_full"}
            for k in ["e1", "e3", "e6", "ef"]:
                vc = VideoClassifier(train_mode="early_fusion", time_step=16, feature_name=arff_paths[k], model_type=mt,
                                     stride=2)
                #vc.print_confusion_matrix("", 2)
                #vc.print_confusion_matrix("", 1)
                del vc
