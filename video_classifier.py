import glob
import random
import sys
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm
from frames_classifier import FramesClassidier
from audio_classifier import AudioClassifier


class VideoClassifier:

    def __init__(self, video_model_path=None, audio_model_path="myModel_17.h5",
                 classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
                 base_path="/user/vlongobardi/AFEW/aligned/"):

        self.fc = FramesClassidier()
        self.ac = AudioClassifier(audio_model_path)
        self.feature_name = "audio_feature_IS09_emotion"  # temporaneo
        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))

        if video_model_path is not None:
            self.model = load_model(video_model_path)
        else:
            skips = 0
            iters = 5
            bs = 16
            ep = 50
            opts = ["Adam", "SGD"]
            lrs = [0.1, 0.01, 0.001, 0.0001]
            self.model = "X"
            self.feature_number = 384
            for opt in opts:
                for lr in lrs:
                    for iteration in range(iters):
                        self.iteration = iteration
                        print(
                            "\n\n################################################################################\n"
                            "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                            " ###########################\n######################################################" +
                            " ########################\nepochs:", ep, "batch_size:", bs,
                            "\nmodel:", "Model" + self.model,  # "in", models,
                            "\nopt:", opt, "in", opts,
                            "\nlr:", lr, "in", lrs)

                        if skips > 0:
                            skips -= 1
                            continue

                        file_name = "videoModel_epoch" + str(ep) + "_lr" + str(lr) + "_Opt" + opt + "_ModelX" + \
                                    "_Feature" + str(self.feature_number) + "_" + str(self.iteration) + ".txt"
                        log_file = open("video_logs/" + file_name, "w")
                        old_stdout = sys.stdout
                        sys.stdout = log_file

                        self.model = self.train_model(base_path + "Train", base_path + "Val", bs, ep, lr, opt)

                        sys.stdout = old_stdout
                        log_file.close()

    def data_gen(self, list_feature_vectors, batch_size, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_feature_vectors)
        while True:
            labels = []
            features = np.zeros((batch_size, 2 * len(self.classes))).astype('float')
            for i in range(c, c + batch_size):
                audio_path = list_feature_vectors[i].split(".")[0].replace("AFEW/aligned", self.feature_name)
                label_from_audio = self.ac.clip_classification(audio_path)
                graund_truth, label_from_frame = self.fc.make_a_prediction(list_feature_vectors[i])
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

    def train_model(self, train_path, val_path, batch_size, epochs, learning_rate, myopt):
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
        train_gen = self.data_gen(train_files, batch_size)
        val_gen = self.data_gen(val_files, batch_size)
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
                     "_lr" + str(learning_rate) + "_Opt" + myopt + "_Model" + str(self.model_number) + \
                     "_Feature" + str(self.feature_number) + "_" + str(self.iteration) + ".h5"

        print("\n\nModels saved as:", model_name)
        print("Train:", history.history['accuracy'][-1], "Val:", history.history['val_accuracy'][-1])
        model.save("video_models/" + model_name)

        return model


vc = VideoClassifier(audio_model_path="audio_models/audioModel_0.23446229100227356_epoch50_lr0.001_OptAdam_Model1_Feature384_1.h5")
