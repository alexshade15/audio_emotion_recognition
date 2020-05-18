from keras.models import Sequential
from keras.layers import Dropout, Dense


def model1(feature_number):  # 6,635
    model = Sequential()
    model.add(Dense(4, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def model2(feature_number):  # 13,583
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def model3(feature_number):  # 26,103
    model = Sequential()
    model.add(Dense(16, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def model4(feature_number):  # 12,943
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model
