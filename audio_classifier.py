import os
import glob
import random
import operator
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm


def get_feature_number(feature_name):
    if feature_name == "audio_feature_IS09_emotion":
        return 384
    if feature_name == "audio_feature_emobase2010":
        return 1582
    return None


def from_arff_to_feture(arff_file):
    with open(arff_file, 'r') as f:
        arff = f.read()
        try:
            header, body = arff.split("@data")
            features = body.split(",")
            features.pop(0)
            features.pop(-1)
        except:
            print("\n\n", arff_file, "\n\n")
    return features


def get_all_arff(path):
    arffs = []
    folders = os.listdir(path)
    for folder in folders:
        sub_arff = os.listdir(path + "/" + folder)
        for arff in sub_arff:
            arffs.append(folder + "/" + arff)
    return arffs


class AudioClassifier:

    def __init__(self, model_path=None, classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
                 base_path="/user/vlongobardi/audio_feature_IS09_emotion/"):

        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))
        if model_path is not None:
            self.model = load_model(model_path)
            self.feature_number = 384  # model_path.split("_")[0] da verificare
        else:
            bs = 16
            ep = 50
            lr = 0.01
            print("epochs:", ep, "batch_size:", bs, "lr:", lr)
            self.feature_number = get_feature_number(base_path.split("/")[-2])
            self.model = self.train_model(base_path + "Train", base_path + "Val", bs, ep, lr, self.feature_number)

    def clip_classification(self, path_clip_beginngin):
        all_predictions = {}
        for c in self.classes:
            all_predictions[c] = 0
        for feature_vector_path in glob.glob(path_clip_beginngin + "*"):
            pred, ground_truth = self.test_model(feature_vector_path)
            all_predictions[pred] += 1
        return max(all_predictions.items(), key=operator.itemgetter(1))[0]

    def test_model(self, sample_path):
        sample = np.array(from_arff_to_feture(sample_path)).reshape(1, self.feature_number)
        ground_truth = sample_path.split("/")[-2]
        prediction = self.model.predict(sample)
        return self.lb.inverse_transform(prediction), ground_truth

    def print_confusion_matrix(self, val_path):
        predictions = []
        ground_truths = []
        stats = []
        for arff in get_all_arff(val_path):
            pred, ground_truth = self.test_model(val_path + arff)
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

    def data_gen(self, feature_folder, list_feature_vectors, batch_size, feature_number=1582, mode="train"):
        c = 0
        if mode == "train":
            random.shuffle(list_feature_vectors)
        while True:
            labels = []
            features = np.zeros((batch_size, feature_number)).astype('float')
            for i in range(c, c + batch_size):
                feature = from_arff_to_feture(feature_folder + "/" + list_feature_vectors[i])
                features[i - c] = np.array(feature)
                labels.append(list_feature_vectors[i].split("/")[0])
            c += batch_size
            if c + batch_size > len(list_feature_vectors):
                c = 0
                random.shuffle(list_feature_vectors)
                if mode == "eval":
                    break
            labels = self.lb.transform(np.array(labels))
            yield features, labels

    def train_model(self, train_path, val_path, batch_size, epochs, learning_rate, feature_number=1582):
        print(feature_number, "\n\n")
        model = Sequential()
        model.add(Dense(16, input_shape=(feature_number,), activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        train_files = get_all_arff(train_path)
        val_files = get_all_arff(val_path)
        train_gen = self.data_gen(train_path, train_files, batch_size, feature_number)
        val_gen = self.data_gen(val_path, val_files, batch_size, feature_number)
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

        model.save("myAudioModel_" + history.history['val_accuracy'][-1] + ".h5")
        return model

# import audio_classifier
# ac = audio_classifier.AudioClassifier("myModel_17.h5")
# ac.print_confusion_matrix("/user/vlongobardi/audio_feature_IS09_emotion/Val/")
