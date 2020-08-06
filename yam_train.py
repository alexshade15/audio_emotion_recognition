import glob
import random
import operator
import librosa
import traceback
import numpy as np

from keras.models import load_model
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from Dataset.Dataset_Utils.dataset_tools import print_cm

from keras_yamnet.yamnet import YAMNet
from keras_yamnet.preprocessing import preprocess_input


def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.1):
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


def augment(x):
    return spec_augment(x, 2)


def get_feature_number(feature_name):
    if "full" in feature_name:
        return 387  # No! Non e costante!
    elif "600" in feature_name:
        return 58
    elif "300" in feature_name:
        return 28
    elif "100" in feature_name:
        return 8
    return None


def get_data_for_generator(base_path="/user/vlongobardi/temp_wav/Train"):
    x = glob.glob(base_path + "/*/*.wav")
    return x  # , [path.split("/")[-2] for path in x]


class YamNetClassifier:
    def __init__(self, model_path=None, classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"],
                 base_path="/user/vlongobardi/IS09_emotion/"):

        self.classes = classes
        self.lb = LabelBinarizer()
        self.lb.fit_transform(np.array(classes))
        self.feature_name = base_path.split("/")[-2]
        self.feature_number = get_feature_number(self.feature_name)
        if model_path is not None:
            self.model = load_model(model_path)
            self.feature_number = get_feature_number(model_path)
        else:
            skips = 0
            iters = 2
            bs = 16  # 128
            ep = 50
            opts = ["SGD", "Adam", "Adagrad"]
            lrs = [0.01, 0.001, 0.0001]  # 0.003
            models = [YAMNet]
            models_name = [x.__name__ for x in models]
            for index, model in enumerate(models):
                for opt in opts:
                    for lr in lrs:
                        for iteration in range(iters):
                            self.iteration = iteration

                            if skips > 0:
                                skips -= 1
                                continue

                            print(
                                "\n\n################################################################################\n"
                                "############################## ITERATION " + str(iteration + 1) + " of " + str(iters) +
                                " ###########################\n######################################################" +
                                " ########################\nepochs:", ep, "batch_size:", bs,
                                "\nModel:", models_name[index], "in", models_name,
                                "\nOpt:", opt, "in", opts,
                                "\nLr:", lr, "in", lrs)

                            self.current_model_name = models_name[index]

                            file_name = "audioModel_epoch" + str(ep) + "_lr" + str(lr) + "_Opt" + opt + "_" + \
                                        models_name[index] + "_Feature" + self.feature_name + "_" + str(
                                self.iteration) + ".txt"
                            # log_file = open("audio_logs/" + file_name, "w")
                            # old_stdout = sys.stdout
                            # sys.stdout = log_file

                            self.model = self.train_model(base_path, bs, ep, lr, opt,
                                                          model(weights='keras_yamnet/yamnet_conv.h5', classes=7,
                                                                classifier_activation='softmax',
                                                                input_shape=(self.feature_number, 64)))
                            # sys.stdout = old_stdout
                            # log_file.close()

    def data_gen(self, list_feature_vectors, batch_size, mode="train", aug=None):
        c = 0
        if mode == "train":
            random.shuffle(list_feature_vectors)
        while True:
            labels = []
            features = np.zeros((batch_size, self.feature_number, 64)).astype('float')
            for i in range(c, c + batch_size):
                try:
                    signal, sound_sr = librosa.load(list_feature_vectors[i], 48000)
                    mel = preprocess_input(signal, sound_sr)
                    mel = mel if aug is None else aug(mel)
                    features[i - c] = np.array(mel)
                    labels.append(list_feature_vectors[i].split("/")[-2])
                except:
                    traceback.print_exc()
                    print("signal shape:", librosa.load(list_feature_vectors[i], 48000)[0].shape)
                    signal, sound_sr = librosa.load(list_feature_vectors[i], 48000)
                    print(preprocess_input(signal, sound_sr).shape)
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
                traceback.print_exc()
                raise Exception('\nLabels!!')
            yield features, labels

    def train_model(self, path, batch_size, epochs, learning_rate=0.1, myopt="Adam", model=None):
        if myopt == "Adam":
            optimizer = Adam(lr=learning_rate)
        elif myopt == "SGD":
            optimizer = SGD(lr=learning_rate)
        else:
            optimizer = Adagrad(lr=learning_rate, decay=1e-6)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_files = get_data_for_generator(path + "Train")
        val_files = get_data_for_generator(path + "Val")

        train_gen = self.data_gen(train_files, batch_size)  # , aug=augment)
        val_gen = self.data_gen(val_files, batch_size)
        no_of_training_images = len(train_files)
        no_of_val_images = len(val_files)

        model_name = "_lr" + str(learning_rate) + "_Opt" + myopt + "_Model" + str(self.current_model_name) + \
                     "_Feature" + self.feature_name + "_" + str(self.iteration) + ".h5"

        cb = [ModelCheckpoint(filepath="audio_models/audioModel_{val_accuracy:.4f}_epoch{epoch:02d}" + model_name,
                              monitor="val_accuracy", save_best_only=True),
              TensorBoard(log_dir="FULL_AUDIO_LOG", write_graph=True, write_images=True)]
        # EarlyStopping(monitor='val_accuracy', patience=10, mode='max')]
        history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=(no_of_training_images // batch_size),
                                      validation_data=val_gen, validation_steps=(no_of_val_images // batch_size),
                                      verbose=1, callbacks=cb)
        # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
        print("\n\nTrain_Accuracy =", history.history['accuracy'])
        print("\nVal_Accuracy =", history.history['val_accuracy'])
        print("\n\nTrain_Loss =", history.history['loss'])
        print("\nVal_Loss =", history.history['val_loss'])

        model_name = "audioModel_" + str(history.history['val_accuracy'][-1]) + "_epoch" + str(epochs) + model_name

        print("\n\nModels saved as:", model_name)
        model.save("audio_models/" + model_name)

        return model

    def clip_classification(self, dataset_path):
        all_predictions = {}
        for c in self.classes:
            all_predictions[c] = 0
        val_files = get_data_for_generator(dataset_path)
        for feature_vector_path in val_files:
            pred, ground_truth = self.test_model(feature_vector_path)
            all_predictions[pred] += 1
        return max(all_predictions.items(), key=operator.itemgetter(1))[0]

    def test_model(self, sample_path):
        signal, sound_sr = librosa.load(sample_path, 48000)
        mel = preprocess_input(signal, sound_sr)
        ground_truth = sample_path.split("/")[-2]
        prediction = self.model.predict(mel)
        return self.lb.inverse_transform(prediction)[0], ground_truth

    def print_confusion_matrix(self, path):
        predictions = []
        ground_truths = []
        stats = []
        for file_path in get_data_for_generator(path + "Val"):
            pred, ground_truth = self.test_model(file_path)
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


if __name__ == "__main__":
    audio_path = {"e1": "emobase2010_100", "e3": "emobase2010_300", "e6": "emobase2010_600", "ef": "emobase2010_full"}
    for e in ["e3", "e1"]:
        ap = "/user/vlongobardi/late_feature/" + audio_path[e] + "_wav/"
        print("######################## AUDIO PATH: ", ap)
        ync = YamNetClassifier(base_path=ap)
