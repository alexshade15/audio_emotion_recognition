import traceback
import glob
import sys
import random
import csv
import pickle
import numpy as np
from os.path import basename, exists

from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
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

        self.fc = FramesClassifier(time_step=time_step)
        self.train_mode = train_mode
        if train_mode == "late_fusion":
            self.ac = AudioClassifier(audio_model_path)
            self.feature_name = '_'.join(audio_model_path.split("Feature")[1].split("_")[0:2])
            print("AC loaded successfully,", audio_model_path, "\nFeature_name:", self.feature_name)
        else:
            self.feature_name = feature_name
        self.feature_number = get_feature_number(self.feature_name)
        self.classes = classes

        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))

        if video_model_path is not None:
            self.model = load_model(video_model_path)
            print("VC loaded successfully", video_model_path)
        else:
            t_files = glob.glob(base_path + "Train" + "/*/*csv")
            v_files = glob.glob(base_path + "Val" + "/*/*csv")
            if self.train_mode == "late_fusion":
                if not exists('lables_late_fusion' + self.feature_name + '.csv'):
                    print("\n##### GENERATING CSV FOR LATE FUSIUON... #####")
                    self.labels_late_fusion = self.generate_data_for_late_fusion(t_files, v_files)
                    print("\n##### CSV GENERATED! #####")
                else:
                    self.labels_late_fusion = {}
                    with open('lables_late_fusion' + self.feature_name + '.csv', 'r') as f:
                        f.readline()
                        csv_reader = csv.reader(f)
                        for row in csv_reader:
                            self.labels_late_fusion[row[0]] = [row[1], row[2], row[3]]
            elif self.train_mode == "early_fusion" and not len(glob.glob("/user/vlongobardi/framefeature_16_50/*/*/*")):
                print("\n##### GENERATING FEATURES FOR EARLY FUSIUON... #####")
                self.generate_feature_for_early_fusion(t_files, v_files, time_step)
                print("\n##### FEATURES GENERATED! #####")

            skips = 0
            iters = 5
            bs = 16
            ep = 50
            opts = ["Adam", "SGD"]
            lrs = [0.1, 0.01, 0.001, 0.0001]
            if train_mode == "late_fusion":
                models = [a_model1, a_model2, a_model3, a_model4, a_model5, a_model5_1, a_model5_2, a_model5_3,
                          a_model6, a_model6_1, a_model6_2, a_model7, a_model7_1]
            else:
                models = [e_model_1_1, e_model_2]
            models_name = [x.__name__ for x in models]
            for index, model in enumerate(models):
                for opt in opts:
                    for lr in lrs:
                        for iteration in range(iters):
                            self.iteration = iteration
                            print(
                                "\n\n################################################################################\n"
                                "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                                " ###########################\n######################################################" +
                                " ########################\nepochs:", ep, "batch_size:", bs,
                                "\nmodel:", models_name[index], "in", models_name,
                                "\nopt:", opt, "in", opts,
                                "\nlr:", lr, "in", lrs)
                            if skips > 0:
                                skips -= 1
                                continue
                            self.current_model_name = models_name[index]
                            # file_name = "videoModel_epoch" + str(ep) + "_lr" + str(lr) + "_Opt" + opt + "_" + \
                            #             models_name[index] + "_Feature" + self.feature_name + "_" + str(
                            #     self.iteration) + "_" + self.train_mode + ".txt"
                            # log_file = open("video_logs/" + file_name, "w")
                            # old_stdout = sys.stdout
                            # sys.stdout = log_file
                            if self.train_mode == "late_fusion":
                                self.model = self.train(t_files, v_files, bs, ep, lr, opt, model(14), self.late_gen)
                            elif self.train_mode == "early_fusion":
                                frame_folder = "framefeature_" + str(time_step) + "_" + str(int(self.fc.overlap * 100))
                                bp = base_path.replace("AFEW/aligned", frame_folder)
                                t_files = glob.glob(bp + "Train" + "/*/*dat")
                                v_files = glob.glob(bp + "Val" + "/*/*dat")
                                self.model = self.train(t_files, v_files, bs, ep, lr, opt, model(self.feature_number),
                                                        self.early_gen)
                            # sys.stdout = old_stdout
                            # log_file.close()

    def generate_data_for_late_fusion(self, train_files, val_files):
        my_csv = {}
        total_files = train_files + val_files
        total = len(total_files)
        for file in total_files:
            clip_id = file.split(".")[0]
            audio_path = clip_id.replace("AFEW/aligned", self.feature_name)
            label_from_audio = self.ac.clip_classification(audio_path)
            graund_truth, label_from_frame = self.fc.predict(file)
            clip_id = basename(clip_id)
            my_csv[clip_id] = [graund_truth, label_from_frame, label_from_audio]
            print(len(my_csv), "/", total)

        with open('lables_late_fusion' + self.feature_name + '.csv', 'w') as f:
            f.write("clip_id, ground_truth, frame_label, audio_label\n")
            for k in my_csv:
                f.write(str(k) + "," + str(my_csv[k][0]) + "," + str(my_csv[k][1]) + "," + str(my_csv[k][2]) + "\n")
        return my_csv

    def generate_feature_for_early_fusion(self, train_files, val_files, time_step):
        self.fc.init_feature_generator()
        video_feature_name = "framefeature_" + str(time_step) + "_" + str(int(self.fc.overlap * 100))
        for file_name in train_files + val_files:
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
                graund_truth, label_from_frame, label_from_audio = self.labels_late_fusion[clip_id]
                features[i - c] = np.append(self.lb.transform(np.array([label_from_audio])),
                                            self.lb.transform(np.array([label_from_frame])))
                labels.append(graund_truth)
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
        time_step = self.fc.time_step
        # new_shape = self.feature_number // 2
        if mode == "train":
            random.shuffle(list_files)
        frame_feature_name = list_files[0].split("/")[3]
        while True:
            labels = []
            features = [np.zeros((batch_size, time_step, 1024)).astype('float'),
                        np.zeros((batch_size, time_step, self.feature_number)).astype('float')]
            for i in range(c, c + batch_size):
                # "/user/vlongobardi/framefeature_16_50/Train/Sad/011603980_0.dat"
                with open(list_files[i], 'rb') as f:
                    features[0][i - c] = pickle.loads(f.read())
                arff_file = list_files[i].replace(frame_feature_name, self.feature_name).replace("dat", "arff")
                arff_feature = np.array(from_arff_to_feture(arff_file)).reshape(1, self.feature_number)
                features[1][i - c] = np.repeat(arff_feature, time_step, axis=0)
                graund_truth = list_files[i].split("/")[-2]
                labels.append(graund_truth)
            c += batch_size
            if c + batch_size > len(list_files):
                c = 0
                random.shuffle(list_files)
                if mode == "eval":
                    break
            labels = np.repeat(self.lb.transform(np.array(labels)).reshape((1, 16, 7)), 16, axis=0)
            yield features, labels

    def train(self, train_files, val_files, batch_size, epochs, learning_rate, myopt, model, generator):
        if myopt == "Adam":
            optimizer = Adam(lr=learning_rate)
        else:
            optimizer = SGD(lr=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_gen = generator(train_files, batch_size)
        val_gen = generator(val_files, batch_size)
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        model_name = "_lr" + str(learning_rate) + "_Opt" + myopt + "_Model" + str(self.current_model_name) + \
                     "_Feature" + self.feature_name + "_" + str(self.iteration) + "_" + self.train_mode + ".h5"

        cb = [ModelCheckpoint(filepath=str("video_models/videoModel_{val_accuracy:.4f}_epoch{epoch:02d}" + model_name),
                              monitor="val_accuracy")]
        # cb.append(TensorBoard(log_dir="logs_audio", write_graph=True, write_images=True))
        history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=(no_of_training_images // batch_size),
                                      validation_data=val_gen, validation_steps=(no_of_val_images // batch_size),
                                      workers=1, verbose=1, callbacks=cb)
        # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
        print("\n\nTrain Accuracy =", history.history['accuracy'])
        print("\nVal Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain Loss =", history.history['loss'])
        print("\nVal Loss =", history.history['val_loss'])

        model_name = "videoModel_" + str(history.history['val_accuracy'][-1]) + "_epoch" + str(epochs) + model_name

        print("\n\nModels saved as:", model_name)
        print("Train:", history.history['accuracy'][-1], "Val:", history.history['val_accuracy'][-1])
        model.save("video_models/" + model_name)

        return model

    # SOLO PER LATE FUSION
    def predict(self, path):
        try:
            audio_path = path.split(".")[0].replace("AFEW/aligned", self.feature_name)
            audio_pred = self.ac.clip_classification(audio_path)
            graund_truth, frame_pred = self.fc.predict(path)
            sample = np.append(self.lb.transform(np.array([audio_pred])), self.lb.transform(np.array([frame_pred])))
            # print("\n\n######## 1", sample.shape, sample)
            pred = self.model.predict(sample.reshape((1, 14)))
        except:
            traceback.print_exc()
            print("\n\n######## 3", sample.shape, sample)
        return self.lb.inverse_transform(pred)[0], audio_pred, frame_pred, graund_truth

    def print_confusion_matrix(self, path):

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
        stats = []
        files = glob.glob(path + "/*/*csv")
        for file in files:
            # print(file)
            id = basename(file).split(".")[0]
            ground_truth, frame_pred, audio_pred = labels_late_fusion[id]
            sample = np.append(self.lb.transform(np.array([audio_pred])), self.lb.transform(np.array([frame_pred])))
            pred = self.model.predict(sample.reshape((1, 14)))
            pred = self.lb.inverse_transform(pred)[0]
            # print(ground_truth, frame_pred, audio_pred, pred)
            # pred, audio_pred, frame_pred, ground_truth = self.predict(file)
            predictions.append(pred)
            a_p.append(audio_pred)
            f_p.append(frame_pred)
            ground_truths.append(ground_truth)

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
        stats = []
        cm = confusion_matrix(ground_truths, a_p, self.classes)
        stats.append(cm)
        stats.append(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2))
        stats.append(accuracy_score(ground_truths, a_p))
        stats.append(classification_report(ground_truths, a_p))

        print("###Audio Results###")
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

        stats = []
        cm = confusion_matrix(ground_truths, f_p, self.classes)
        stats.append(cm)
        stats.append(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2))
        stats.append(accuracy_score(ground_truths, f_p))
        stats.append(classification_report(ground_truths, f_p))

        print("###Frame Results###")
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


def test_51(ka):
    for i in range(ka):
        model_path = "audio_models/audioModel_0.2953_epoch14_lr0.001_OptAdam_Modela_model6_1_Featureemobase2010_600_0.h5"
        vc = VideoClassifier(train_mode="late_fusion", audio_model_path=model_path)
        vc.print_confusion_matrix("/user/vlongobardi/AFEW/aligned/Val")


if __name__ == "__main__":
    if sys.argv[1] == "late":
        print("LATE")
        # model_path = "audio_models/audioModel_0.2953_epoch14_lr0.001_OptAdam_Modela_model6_1_Featureemobase2010_600_0.h5"
        model_path = "audio_models/audioModel_0.3668_epoch37_lr0.001_OptSGD_Modela_model7_Featureemobase2010_full_1.h5"
        vc = VideoClassifier(train_mode="late_fusion", audio_model_path=model_path)
        # vc.print_confusion_matrix("/user/vlongobardi/AFEW/aligned/Val")
    else:
        print("EARLY")
        arff_paths = {"e": "emobase2010_300", "i": "IS09_emotion_300"}
        arff_path = arff_paths[sys.argv[2]]
        vc = VideoClassifier(train_mode="early_fusion", feature_name=arff_path)
