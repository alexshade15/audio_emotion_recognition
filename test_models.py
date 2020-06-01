from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, TimeDistributed, Input, Concatenate


def a_model1(feature_number=384):  # 6,635 // 1,843
    model = Sequential()
    model.add(Dense(4, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model2(feature_number=384):  # 13,583 // 3,999
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model3(feature_number=384):  # 26,103 // 6,935
    model = Sequential()
    model.add(Dense(16, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model4(feature_number=384):  # 12,943 // 3,359
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model5(feature_number=384):  # 104,119 // 27,447
    model = Sequential()
    model.add(Dense(64, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model5_1(feature_number=384):  # 102,759 // 27,367
    model = Sequential()
    model.add(Dense(64, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model5_2(feature_number=384):  # 51,303 // 12,967
    model = Sequential()
    model.add(Dense(32, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model5_3(feature_number=384):
    model = Sequential()
    model.add(Dense(16, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model6(feature_number=384):  # 15,759 // 6,335
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model6_1(feature_number=384):  # 24,687 // 15,103
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def a_model6_2(feature_number=384):  # 58,607 // 49,023
    model = Sequential()
    model.add(Dense(8, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def v_model_X(feature_number=14):
    model = Sequential()
    model.add(Dense(16, input_shape=(feature_number,), activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model


def e_model_1(feature_number=384):
    # audio_input = Input(shape=(2, feature_number // 2))    16?
    audio_input = Input(shape=(16, feature_number))
    frame_input = Input(shape=(16, 1024))
    combined = Concatenate([frame_input, audio_input])
    # kernel_regularizer=regularizers.l2(weight_decay)
    x = TimeDistributed(Dense(100, activation='tanh'))(combined)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(7, activation='softmax'))(x)
    return Model(inputs=[frame_input, audio_input], outputs=x)




