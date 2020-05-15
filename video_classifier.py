import frames_classifier
import audio_classifier

import os
import glob
import random
import operator
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
        self.feature_number = 384  # temporaneo
        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))

        if video_model_path is not None:
            self.model = load_model(video_model_path)
        else:
            batch_size = 16
            epeoch = 50
            lr = 0.01
            print("epochs:", epeoch, "batch_size:", batch_size, "lr:", lr)
            self.model = self.train_model(base_path + "Train", base_path + "Val", batch_size, epeoch, lr)

    def data_gen(self, list_feature_vectors, batch_size, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_feature_vectors)
        while True:
            labels = []
            features = np.zeros((batch_size, 2*len(self.classes))).astype('float')
            for i in range(c, c + batch_size):
                label_from_audio = self.ac.clip_classification(list_feature_vectors[i].split(".")[0])
                graund_truth, label_from_frame = self.fc.make_a_prediction(list_feature_vectors[i])
                print("\n\n\nlabel_from_audio, label_from_frame, graund_truth: ", label_from_audio, label_from_frame, graund_truth)
                features[i - c] = np.append(self.lb.transform(label_from_audio), self.lb.transform(label_from_frame))
                labels.append(graund_truth)
            c += batch_size
            if c + batch_size > len(list_feature_vectors):
                c = 0
                random.shuffle(list_feature_vectors)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels))
            print("\n\n\nfeatures, labels: ", features, labels)
            yield features, labels

    def train_model(self, train_path, val_path, batch_size, epochs, learning_rate):
        model = Sequential()
        model.add(Dense(16, input_shape=(2 * len(self.classes),), activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        train_files = glob.glob(train_path + "/*/*csv")
        val_files = glob.glob(val_path + "/*/*csv")
        train_gen = self.data_gen(train_files, batch_size)
        val_gen = self.data_gen(val_files, batch_size)
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        # tb_call_back = TensorBoard(log_dir="logs_audio", write_graph=True, write_images=True)
        history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=(no_of_training_images // batch_size),
                                      validation_data=val_gen, validation_steps=(no_of_val_images // batch_size))
        #                              callbacks=[tb_call_back])
        # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
        print("\n\nTrain Accuracy =", history.history['accuracy'])
        print("\nVal Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain Loss =", history.history['loss'])
        print("\nVal Loss =", history.history['val_loss'])

        model.save("myVideoModel_" + history.history['val_accuracy'][-1] + ".h5")
        return model
