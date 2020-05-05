import os
import csv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import TensorBoard


def get_labels(path):
    csvfile = open(path, 'rb')
    reader = csv.reader(csvfile)
    next(reader, None)

    rows = []
    for row in reader:
        rows.append(row)
    return np.array(rows)


def from_arff_to_feture(arff_file):
    with open(arff_file, 'r') as f:
        arff = f.read()
        header, body = arff.split("@data")
        features = body.split(",")
        features.pop(0)  # remove name
        features.pop(0)  # remove frameIndex
        features.pop(0)  # remove frameTime
        features.pop(-1)
    return features


def get_all_arff(path):
    arffs = []
    dirs = os.listdir(path)
    for dir in dirs:
        sub_arff = os.listdir(path + "/" + dir)
        for arff in sub_arff:
            arffs.append(dir + "/" + arff)
    return arffs


def data_gen(feature_folder, batch_size, mode="train"):
    lbs = ["Angry", "Disgust"  "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    lb = LabelBinarizer()
    lb.fit(lbs)
    c = 0
    n1 = get_all_arff(feature_folder)  # List of training feature vector
    if mode == "train":
        random.shuffle(n1)
    while True:
        labels = []
        features = np.zeros((batch_size, 1582, 1)).astype('float')
        for i in range(c, c + batch_size):
            features[i - c] = from_arff_to_feture(feature_folder + "/" + n1[i])
            labels.append(n1[i].split("/")[0])
        c += batch_size
        if c + batch_size - 1 > len(n1):
            c = 0
            random.shuffle(n1)
            if mode == "eval":
                break
        labels = lb.transform(np.array(labels))
        yield features, labels


def train_model(train_path, val_path, batch_size, epochs):
    model = Sequential()
    model.add(Dense(128, input_shape=(batch_size, 1582, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_gen = data_gen(train_path, batch_size=batch_size)
    val_gen = data_gen(val_path, batch_size=batch_size)
    no_of_training_images = len(os.listdir(train_path))
    no_of_val_images = len(os.listdir(val_path))

    tb_call_back = TensorBoard(log_dir="audio_logs", write_graph=True, write_images=True)
    history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=(no_of_training_images // batch_size),
                                  validation_data=val_gen, validation_steps=(no_of_val_images // batch_size))
    # score = model.evaluate_generator(test_gen, no_of_test_images // batch_size)
