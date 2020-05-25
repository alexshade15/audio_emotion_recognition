import glob
import random
import sys
import csv
import numpy as np
from os.path import basename, exists

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm
from frames_classifier import FramesClassidier
from audio_classifier import AudioClassifier


def get_feature_name(feature_number):
    if feature_number == 384:
        return "audio_feature_IS09_emotion"
    if feature_number == 1582:
        return "audio_feature_emobase2010"
    return None


class VideoClassifier:

    def __init__(self, train_mode="late_fusion", video_model_path=None, audio_model_path="myModel_17.h5",
                 classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
                 base_path="/user/vlongobardi/AFEW/aligned/"):

        self.fc = FramesClassidier()
        self.ac = AudioClassifier(audio_model_path)
        self.feature_number = int(audio_model_path.split("Feature")[1].split("_")[0])
        self.feature_name = get_feature_name(self.feature_number)
        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))

        if video_model_path is not None:
            self.model = load_model(video_model_path)
        else:

            if train_mode == "late_fusion" and \
                    not exists('lables_late_fusion' + self.feature_name.replace("audio_feature", "") + '.csv'):
                print("\n##### GENERATING CSV FOR LATE FUSIUON... #####")
                self.labels_late_fusion = self.generate_data_for_late_fusion(base_path)
                print("\n##### CSV GENERATED! #####")
            else:
                self.labels_late_fusion = {}
                with open('lables_late_fusion' + self.feature_name.replace("audio_feature", "") + '.csv', 'r') as f:
                    f.readline()
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        self.labels_late_fusion[row[0]] = [row[1], row[2], row[3]]

            skips = 0
            iters = 5
            bs = 16
            ep = 50
            opts = ["Adam", "SGD"]
            lrs = [0.1, 0.01, 0.001, 0.0001]
            self.model_name = "X"
            for opt in opts:
                for lr in lrs:
                    for iteration in range(iters):
                        self.iteration = iteration
                        print(
                            "\n\n################################################################################\n"
                            "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                            " ###########################\n######################################################" +
                            " ########################\nepochs:", ep, "batch_size:", bs,
                            "\nmodel:", "Model" + self.model_name,  # "in", models,
                            "\nopt:", opt, "in", opts,
                            "\nlr:", lr, "in", lrs)

                        if skips > 0:
                            skips -= 1
                            continue

                        file_name = "videoModel_epoch" + str(ep) + "_lr" + str(lr) + "_Opt" + opt + "_ModelX" + \
                                    "_Feature" + str(self.feature_number) + "_" + str(self.iteration) + ".txt"
                        # log_file = open("video_logs/" + file_name, "w")
                        # old_stdout = sys.stdout
                        # sys.stdout = log_file
                        if train_mode == "late_fusion":
                            self.model = self.late_training(base_path + "Train", base_path + "Val", bs, ep, lr, opt)
                        elif train_mode == "early_fusion":
                            self.model = self.training(base_path + "Train", base_path + "Val", bs, ep, lr, opt)
                        elif train_mode == "train_level":
                            self.model = self.training(base_path + "Train", base_path + "Val", bs, ep, lr, opt)

                        # sys.stdout = old_stdout
                        # log_file.close()

    def generate_data_for_late_fusion(self, base_path):
        train_path = base_path + "Train"
        val_path = base_path + "Val"

        train_files = glob.glob(train_path + "/*/*csv")
        train_files.remove("/user/vlongobardi/AFEW/aligned/Train/Surprise/011603980.csv")
        train_files.remove("/user/vlongobardi/AFEW/aligned/Train/Angry/004510640.csv")
        train_files.remove("/user/vlongobardi/AFEW/aligned/Train/Sad/001821420.csv")
        val_files = glob.glob(val_path + "/*/*csv")

        my_csv = {}
        for file in train_files + val_files:
            clip_id = file.split(".")[0]
            audio_path = clip_id.replace("AFEW/aligned", self.feature_name)
            label_from_audio = self.ac.clip_classification(audio_path)
            graund_truth, label_from_frame = self.fc.predict(file)
            clip_id = basename(clip_id)
            my_csv[clip_id] = [graund_truth, label_from_frame, label_from_audio]

        with open('lables_late_fusion' + self.feature_name.replace("audio_feature", "") + '.csv', 'w') as f:
            f.write("clip_id, ground_truth, frame_label, audio_label\n")
            for k in my_csv:
                f.write(str(k) + ", " + str(my_csv[k][0]) + ", " + str(my_csv[k][1]) + ", " + str(my_csv[k][2]) + "\n")
        return my_csv

    def late_data_gen(self, list_feature_vectors, batch_size, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_feature_vectors)
        while True:
            labels = []
            features = np.zeros((batch_size, 2 * len(self.classes))).astype('float')
            for i in range(c, c + batch_size):
                clip_id = basename(list_feature_vectors[i].split(".")[0])
                graund_truth, label_from_frame, label_from_audio = self.labels_late_fusion[clip_id]

                # audio_path = list_feature_vectors[i].split(".")[0].replace("AFEW/aligned", self.feature_name)
                # label_from_audio = self.ac.clip_classification(audio_path)
                # graund_truth, label_from_frame = self.fc.predict(list_feature_vectors[i])
                features[i - c] = np.append(self.lb.transform(np.array([label_from_audio])),
                                            self.lb.transform(np.array([label_from_frame])))
                labels.append(graund_truth)
            c += batch_size
            if c + batch_size > len(list_feature_vectors):
                c = 0
                random.shuffle(list_feature_vectors)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels))
            # print("\n\n\n#######features, labels: ", features.shape, labels.shape)
            yield features, labels

    def late_training(self, train_path, val_path, batch_size, epochs, learning_rate, myopt):
        model = Sequential()
        model.add(Dense(16, input_shape=(2 * len(self.classes),), activation='relu'))
        model.add(Dense(7, activation='softmax'))

        if myopt == "Adam":
            optimizer = Adam(lr=learning_rate)
        else:
            optimizer = SGD(lr=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_files = glob.glob(train_path + "/*/*csv")
        train_files.remove("/user/vlongobardi/AFEW/aligned/Train/Surprise/011603980.csv")
        train_files.remove("/user/vlongobardi/AFEW/aligned/Train/Angry/004510640.csv")
        train_files.remove("/user/vlongobardi/AFEW/aligned/Train/Sad/001821420.csv")
        val_files = glob.glob(val_path + "/*/*csv")
        train_gen = self.late_data_gen(train_files, batch_size)
        val_gen = self.late_data_gen(val_files, batch_size)
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        # tb_call_back = TensorBoard(log_dir="logs_audio", write_graph=True, write_images=True)
        history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=(no_of_training_images // batch_size),
                                      validation_data=val_gen, validation_steps=(no_of_val_images // batch_size),
                                      workers=0, verbose=0)
        #                              callbacks=[tb_call_back])
        # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
        print("\n\nTrain Accuracy =", history.history['accuracy'])
        print("\nVal Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain Loss =", history.history['loss'])
        print("\nVal Loss =", history.history['val_loss'])

        model_name = "videoModel_" + str(history.history['val_accuracy'][-1]) + "_epoch" + str(epochs) + \
                     "_lr" + str(learning_rate) + "_Opt" + myopt + "_Model" + str(self.model_name) + \
                     "_Feature" + str(self.feature_number) + "_" + str(self.iteration) + ".h5"

        print("\n\nModels saved as:", model_name)
        print("Train:", history.history['accuracy'][-1], "Val:", history.history['val_accuracy'][-1])
        model.save("video_models/" + model_name)

        return model

    def early_training(self):
        # inquadrare l'architettura video
        # rimuovere dall'architettura video i layers di classificazione, creando un modello in grado di generare feature
        # creare una nuova architettura che utilizza le feature video più quelle audio per classificare
        # classiifcazione su l'intero video o su spezzoni? #SPEZZONI: lunghezza seq video compatibile con window audio
        # AUG mixando video e audio
        pass

    def train_training(self):
        pass

    def predict(self, path):
        audio_path = path.split(".")[0].replace("AFEW/aligned", self.feature_name)
        audio_pred = self.ac.clip_classification(audio_path)
        graund_truth, frame_pred = self.fc.predict(path)
        sample = np.append(self.lb.transform(np.array([audio_pred])), self.lb.transform(np.array([frame_pred])))
        pred = self.model.predict(sample)
        return self.lb.inverse_transform(pred)[0], graund_truth

    def print_confusion_matrix(self, path):
        predictions = []
        ground_truths = []
        stats = []
        files = glob.glob(path + "/*/*csv")
        for file in files:
            pred, ground_truth = self.predict(file)
            predictions.append(pred)
            ground_truths.append(ground_truth)

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


vc = VideoClassifier(
    audio_model_path="audio_models/audioModel_0.23446229100227356_epoch50_lr0.001_OptAdam_Model1_Feature384_1.h5")
