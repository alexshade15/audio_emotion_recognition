import os

from keras import Model, Input, regularizers
from keras.layers import TimeDistributed, LSTMCell, Reshape, Dense, Lambda, Dropout, Concatenate
import tensorflow as tf
from Models.RNN_stacked_attention import RNNStackedAttention
from Models.seresnet50 import SEResNet50

from keras_yamnet.yamnet import YAMNet


#
# Original implementation:
# https://github.com/kracwarlock/action-recognition-visual-attention/blob/6738a0e2240df45ba79e87d24a174f53adb4f29b/src/actrec.py#L111
# Timestep: 30
#

def SharmaNet(input_shape, train_all_baseline=False, weight_decay=1e-5, train_mode="default", audio_dim=0, yam_dim=0):
    frame_input = Input(input_shape)
    regs = [regularizers.l2(weight_decay), regularizers.l2(weight_decay), regularizers.l2(weight_decay),
            regularizers.l2(weight_decay), regularizers.l2(weight_decay), regularizers.l2(weight_decay)]
    cells = [LSTMCell(1024, kernel_regularizer=regs[0], recurrent_regularizer=regs[1])]

    seres50 = SEResNet50(input_shape=(input_shape[1], input_shape[2], input_shape[3]), classes=7)
    backbone = Model(seres50.input, seres50.layers[-4].output)
    x = TimeDistributed(backbone, input_shape=input_shape)(frame_input)

    t, h, w, c = [int(x) for x in x.shape[1:]]
    reshape_dim = (t, h * w, c)
    x = Reshape(reshape_dim)(x)

    features_mean_layer = Lambda(lambda y: tf.reduce_mean(y, axis=2))(x)
    features_mean_layer = Lambda(lambda y: tf.reduce_mean(y, axis=1))(features_mean_layer)

    dense_h0 = Dense(1024, activation='tanh', kernel_regularizer=regs[2])(features_mean_layer)
    dense_c0 = Dense(1024, activation='tanh', kernel_regularizer=regs[3])(features_mean_layer)

    rnn_attention = RNNStackedAttention(reshape_dim, cells, return_sequences=True, unroll=True)
    x = rnn_attention(x, initial_state=[dense_h0, dense_c0])

    if train_mode != "default":
        if "early" in train_mode:
            audio_input = Input(shape=(input_shape[0], audio_dim))
        elif "joint" in train_mode:
            yn = YAMNet(weights='keras_yamnet/yamnet_conv.h5', classes=7, classifier_activation='softmax',
                        input_shape=(yam_dim, 64))
            yamnet = Model(input=yn.input, output=yn.layers[-3].output)
            audio_input = yamnet.output

        x = Concatenate(name='fusion1')([x, audio_input])
        input_tensors = [audio_input, frame_input]
    else:
        input_tensors = frame_input

    x = TimeDistributed(Dense(100, activation='tanh', kernel_regularizer=regs[4], name='ff_logit_lstm'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(7, activation='softmax', kernel_regularizer=regs[5], name='ff_logit'))(x)
    x = Lambda(lambda y: tf.reduce_mean(y, axis=1))(x)

    model = Model(input_tensors, x)
    model.layers[1].trainable = train_all_baseline

    return model


if __name__ == "__main__":
    i = Input((16, 224, 224, 3))
    m = SharmaNet(i)
    print(m.summary())
