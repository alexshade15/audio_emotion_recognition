import os

from keras import Model, Input, regularizers
from keras.layers import TimeDistributed, LSTMCell, Reshape, Dense, Lambda, Dropout, Concatenate
import tensorflow as tf
from Models.RNN_stacked_attention import RNNStackedAttention
from Models.seresnet50 import SEResNet50

#
# Original implementation:
# https://github.com/kracwarlock/action-recognition-visual-attention/blob/6738a0e2240df45ba79e87d24a174f53adb4f29b/src/actrec.py#L111
# Timestep: 30
# 
#
def SharmaNet(input_shape,train_all_baseline=False, classification = True, weight_decay=1e-5, weights = 'afew', train_mode="default", audio_dim=None):
    input = Input(input_shape)
    cells = [LSTMCell(1024,kernel_regularizer=regularizers.l2(weight_decay),recurrent_regularizer=regularizers.l2(weight_decay))]



    seres50 = SEResNet50(input_shape=(input._keras_shape[2],input._keras_shape[3],input._keras_shape[4]), classes=7)
    # load FER weights
    # pop the last classification layers
    seres50.summary()
    backbone = Model(seres50.input, seres50.layers[-4].output)
    x = TimeDistributed(backbone, input_shape=(input._keras_shape))(input)

    T, H, W, C = [int(x) for x in x.shape[1:]]
    reshape_dim = (T, H * W, C)
    x = Reshape((reshape_dim))(x)

    #mlp for lstm initialization
    features_mean_layer =  Lambda(lambda y: tf.reduce_mean(y, axis=2))(x)
    features_mean_layer =  Lambda(lambda y: tf.reduce_mean(y, axis=1))(features_mean_layer)

    dense_h0 = Dense(1024,activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(features_mean_layer)
    dense_c0 = Dense(1024,activation='tanh', kernel_regularizer=regularizers.l2(weight_decay))(features_mean_layer)


    Rnn_attention = RNNStackedAttention(reshape_dim, cells, return_sequences=True, unroll=True)

    if "early" in train_mode:
        audio_input = Input(shape=(input._keras_shape[0], audio_dim))
        x = Concatenate(name='fusion1')([x, audio_input])

    x = Rnn_attention(x,initial_state=[dense_h0,dense_c0])  # (BS,TS,lstm_out)
    x = TimeDistributed(Dense(100, activation='tanh',kernel_regularizer=regularizers.l2(weight_decay), name='ff_logit_lstm'))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    if classification:
        n_classes = 7
        x = TimeDistributed(Dense(n_classes, activation='softmax',kernel_regularizer=regularizers.l2(weight_decay), name='ff_logit'))(x) # line 405
    else:
        n_outs = 2
        x = Dense(n_outs,activation='tanh',kernel_regularizer=regularizers.l2(weight_decay))(x)
    
    x = Lambda(lambda y: tf.reduce_mean(y, axis=1))(x)

    if audio_dim is not None:
        input_tensors = [audio_input, input]
    else:
        input_tensors = input

    model = Model(input_tensors, x)
    model.layers[1].trainable = train_all_baseline

    return model

if __name__ == "__main__":
    x = Input((16,224,224,3))
    model =SharmaNet(x)
    print(model.summary())