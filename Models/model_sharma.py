from tensorflow.compat.v1 import enable_eager_execution
#enable_eager_execution()
import os
import tensorflow as tf
from keras import Model, Input, regularizers
from keras.layers import TimeDistributed, LSTMCell, Reshape, Dense, Lambda, Dropout
from Models.RNN_stacked_attention import RNNStackedAttention
from Models.seresnet50 import SEResNet50

#
# Original implementation:
# https://github.com/kracwarlock/action-recognition-visual-attention/blob/6738a0e2240df45ba79e87d24a174f53adb4f29b/src/actrec.py#L111
# Timestep: 30
#
#

basepath = "/user/vlongobardi/"


def SharmaNet(input_shape, train_all_baseline=False, classification=True, weight_decay=1e-5, weights='afew', dim=0):
    if dim > 2:
        cell_dim = 1024
    else:
        cell_dim = 3630
    cells = [LSTMCell(cell_dim, kernel_regularizer=regularizers.l2(weight_decay),
                      recurrent_regularizer=regularizers.l2(weight_decay))]

    input_layer = Input(input_shape)
    # print(basepath)
    # create instance of SEesNet50, num classes is num_classes+1, weight trained on FER (8 classes), AFEW 7
    if weights == 'afew':
        print("afew weights")
        weights_path = os.path.join(basepath, "SENET_best_checkpoint_AFEW.hdf5")
        classes = 7
    if weights == 'raf':
        # print("raf weights")
        weights_path = os.path.join(basepath, "raf_wd005_mom_lpf0_ep195.hdf5")
        classes = 7
    if weights == 'aff':
        weights_path = os.path.join(basepath, "SENET50_AFF_from_RAF_1-4_64.hdf5")
        # print("aff weights")
        classes = 2
    if weights == 'recola':
        weights_path = os.path.join(basepath, "SENET_50_RECOLA_from_RAF.hdf5")
        # print("recola weights")
        classes = 2
    #print("SEResNet50(input_shape:", input_shape[1:], (None, input_shape[1:]))
    seres50 = SEResNet50(input_shape=input_shape[1:], classes=classes)
    # load FER weights

    seres50.load_weights(weights_path)
    # pop the last classification layers
    # seres50.summary()
    backbone = Model(seres50.input, seres50.layers[-4].output)
    x = TimeDistributed(backbone, input_shape=(input_shape))(input_layer)

    T, H, W, C = [int(x) for x in x.shape[1:]]
    reshape_dim = (T, H * W, C)
    x = Reshape(reshape_dim)(x)

    # mlp for lstm initialization
    features_mean_layer = Lambda(lambda y: tf.reduce_mean(y, axis=2))(x)
    features_mean_layer = Lambda(lambda y: tf.reduce_mean(y, axis=1))(features_mean_layer)

    dense_h0 = Dense(cell_dim, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(features_mean_layer)
    dense_c0 = Dense(cell_dim, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(features_mean_layer)

    #audio_input = Input(shape=(1582,))
    Rnn_attention = RNNStackedAttention(reshape_dim, cells, return_sequences=True, unroll=True)
    x = Rnn_attention(x, initial_state=[dense_h0, dense_c0])  # (BS,TS,lstm_out)
    x = TimeDistributed(
        Dense(100, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay), name='ff_logit_lstm'))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    if classification:
        n_classes = 7
        x = TimeDistributed(
            Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay), name='ff_logit'))(
            x)  # line 405
    else:
        n_outs = 2
        x = Dense(n_outs, activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Lambda(lambda y: tf.reduce_mean(y, axis=1))(x)

    input_tensors = Rnn_attention.get_audio_tensors()
    # print("Rnn_attention, audio_tensors:", len(input_tensors))
    # print(input_tensors)

    input_tensors.append(input_layer)

    model = Model(input_tensors, x)
    model.layers[1].trainable = train_all_baseline

    return model


if __name__ == "__main__":
    i = Input((16, 224, 224, 3))
    in_sh = (16, 224, 224, 3)
    model = SharmaNet(in_sh)
    print(model.summary())
