
import os
import csv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD


def get_feature_number(feature_name):
    if feature_name == "audio_feature_IS09_emotion":
        return 384
    if feature_name == "audio_feature_emobase2010":
        return 1582
    return None


def get_labels(path):
    csvfile = open(path, 'rb')
    reader = csv.reader(csvfile)
    next(reader, None)

    rows = []
    for row in reader:
        rows.append(row)
    return np.array(rows)


def from_arff_to_feture(arff_file):
    # print("\n\n\narff_file", arff_file)
    with open(arff_file, 'r') as f:
        arff = f.read()
        # print("\n\n\narff", arff)
        header, body = arff.split("@data")
        # print("\n\n\nbody", body.encode('utf-8'), "--")
        try:
            features = body.split(",")
            features.pop(0)  # remove name
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


def data_gen(feature_folder, list_feature_vectors, batch_size, feature_number=1582, mode="train"):
    lbs = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    lb = LabelBinarizer()
    lb.fit_transform(np.array(lbs))
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
            except:
                print("\n\n\n", len(list_feature_vectors), i, c, batch_size)
                print("\n\n\n\n\n", c, batch_size, len(list_feature_vectors), c + batch_size, "\n\n\n\n\n\n\n")
                print(feature_folder + "/" + list_feature_vectors[i], "\n\n")
            labels.append(list_feature_vectors[i].split("/")[0])
        c += batch_size
        # print("\n\n\n\n\n", c, batch_size, len(list_feature_vectors), c + batch_size, "\n\n\n\n\n\n\n")
        if c + batch_size > len(list_feature_vectors):
            c = 0
            random.shuffle(list_feature_vectors)
            if mode == "eval":
                break
        # print("\n\nLABEL", labels)
        labels = lb.transform(np.array(labels))
        # print("\n\nLABEL", labels)
        yield features, labels


def train_model(train_path, val_path, batch_size, epochs, learning_rate, feature_number=1582):
    print(feature_number, "\n\n")
    model = Sequential()
    model.add(Dense(16, input_shape=(feature_number,), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    train_files = get_all_arff(train_path)
    val_files = get_all_arff(val_path)
    train_gen = data_gen(train_path, train_files, batch_size, feature_number)
    val_gen = data_gen(val_path, val_files, batch_size, feature_number)
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


bs = 16
ep = 50
lr = 0.01
base_path = "/user/vlongobardi/audio_feature_IS09_emotion/"
print("epochs:", ep, "batch_size:", bs, "lr:", lr)
train_model(base_path + "Train", base_path + "Val", bs, ep, lr, get_feature_number(base_path.split("/")[-2]))
