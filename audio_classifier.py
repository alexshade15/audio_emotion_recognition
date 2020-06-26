import sys
import os
import glob
import random
import operator
import numpy as np

from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm
from test_models import *


def get_feature_number(feature_name):
    if "IS09_emotion" in feature_name:
        return 384
    if "emobase2010" in feature_name:
        return 1582
    return None


def from_arff_to_feture(arff_file):
    try:
        with open(arff_file, 'r') as f:
            arff = f.read()
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
                 base_path="/user/vlongobardi/IS09_emotion/"):

        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))
        if model_path is not None:
            self.model = load_model(model_path)
            # self.feature_number = int(model_path.split("_Feature")[-1].split("_")[0])
            self.feature_number = get_feature_number(model_path.split("_Feature")[-1].split("_")[0])
        else:
            # fine tuning
            skips = 0
            iters = 10
            bs = 16
            ep = 50
            opts = ["Adam", "SGD"]
            lrs = [0.01, 0.001, 0.0001]  # 0.1, 0.01, 0.001, 0.0001]
            # models = [a_model1, a_model2, a_model3, a_model4, a_model5, a_model5_1, a_model5_2, a_model5_3, a_model6, a_model6_1, a_model6_2, a_model7, a_model7_1]
            models = [a_model1, a_model5, a_model5_3, a_model6, a_model6_1, a_model6_2, a_model_7, a_model_7_1]
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
                                "\nModel:", models_name[index], "in", models_name,
                                "\nOpt:", opt, "in", opts,
                                "\nLr:", lr, "in", lrs)

                            if skips > 0:
                                skips -= 1
                                continue
                            self.current_model_name = models_name[index]
                            self.feature_name = base_path.split("/")[-2]
                            self.feature_number = get_feature_number(self.feature_name)

                            file_name = "audioModel_epoch" + str(ep) + "_lr" + str(lr) + "_Opt" + opt + "_" + \
                                        models_name[index] + "_Feature" + self.feature_name + "_" + str(
                                self.iteration) + ".txt"
                            # log_file = open("audio_logs/" + file_name, "w")
                            # old_stdout = sys.stdout
                            # sys.stdout = log_file

                            self.model = self.train_model(base_path + "Train", base_path + "Val", bs, ep, lr, opt,
                                                          model(self.feature_number))
                            # sys.stdout = old_stdout
                            # log_file.close()

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
        return self.lb.inverse_transform(prediction)[0], ground_truth

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
                try:
                    feature = from_arff_to_feture(feature_folder + "/" + list_feature_vectors[i])
                    features[i - c] = np.array(feature)
                    labels.append(list_feature_vectors[i].split("/")[0])
                except:
                    print("\n\ni:", i, "\nc:", c, "\nist_feature_vectors[i]:", list_feature_vectors[i])
            c += batch_size
            if c + batch_size > len(list_feature_vectors):
                c = 0
                random.shuffle(list_feature_vectors)
                if mode == "eval":
                    break
            try:
                labels = self.lb.transform(np.array(labels))
            except:
                print("\n\n#############", labels)
                print("\n\n#############", np.array(labels))
                print("\n\nc:", c, "\nist_feature_vectors[i]:", list_feature_vectors[c:c + batch_size])
                raise Exception('\nLabels!!')
            yield features, labels

    def train_model(self, train_path, val_path, batch_size, epochs, learning_rate=0.1, myopt="Adam", model=None):
        if myopt == "Adam":
            optimizer = Adam(lr=learning_rate)
        else:
            optimizer = SGD(lr=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_files = get_all_arff(train_path)
        val_files = get_all_arff(val_path)
        train_gen = self.data_gen(train_path, train_files, batch_size, self.feature_number)
        val_gen = self.data_gen(val_path, val_files, batch_size, self.feature_number)
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        model_name = "_lr" + str(learning_rate) + "_Opt" + myopt + "_Model" + str(self.current_model_name) + \
                     "_Feature" + self.feature_name + "_" + str(self.iteration) + ".h5"

        cb = [ModelCheckpoint(filepath="audio_models/audioModel_{val_accuracy:.4f}_epoch{epoch:02d}" + model_name,
                              monitor="val_accuracy")]
        # cb.append(TensorBoard(log_dir="logs_audio", write_graph=True, write_images=True))
        history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=(no_of_training_images // batch_size),
                                      validation_data=val_gen, validation_steps=(no_of_val_images // batch_size),
                                      verbose=1, callbacks=cb)
        # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
        print("\n\nTrain Accuracy =", history.history['accuracy'])
        print("\nVal Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain Loss =", history.history['loss'])
        print("\nVal Loss =", history.history['val_loss'])

        model_name = "audioModel_" + str(history.history['val_accuracy'][-1]) + "_epoch" + str(epochs) + model_name

        print("\n\nModels saved as:", model_name)
        model.save("audio_models/" + model_name)

        return model


if __name__ == "__main__":
    try:
        audio_path = sys.argv[1]
    except:
        audio_path = "/user/vlongobardi/emobase2010_300/"
    print("######################## AUDIO PATH: ", audio_path)
    ac = AudioClassifier(base_path=audio_path)
