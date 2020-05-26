import os

from keras import Model, Input, regularizers
from keras.layers import TimeDistributed, LSTMCell, Reshape, Dense, Lambda, Dropout
import tensorflow as tf
from Models.RNN_stacked_attention import RNNStackedAttention
from Models.seresnet50 import SEResNet50

#
# Original implementation:
# https://github.com/kracwarlock/action-recognition-visual-attention/blob/6738a0e2240df45ba79e87d24a174f53adb4f29b/src/actrec.py#L111
# Timestep: 30
#
#

basepath = "/user/rpalladino/Models/Weights"

def SharmaNet(input,train_all_baseline=False, classification = True, weight_decay=1e-5, weights = 'afew'):
    
    cells = [LSTMCell(1024,kernel_regularizer=regularizers.l2(weight_decay),recurrent_regularizer=regularizers.l2(weight_decay))]
    #print(basepath)
    # # create instance of SEesNet50, num classes will be num_classes+1 because the weight to load were trained on FER (8 classes), AFEW is 7
    if weights == 'afew':
        #print("afew weights")
        weights_path = os.path.join(basepath,"SENET_best_checkpoint_AFEW.hdf5")
        classes = 7
    if weights == 'raf':
        #print("raf weights")
        weights_path = os.path.join(basepath,"raf_wd005_mom_lpf0_ep195.hdf5")
        classes = 7
    if weights == 'aff':
        weights_path = os.path.join(basepath,"SENET50_AFF_from_RAF_1-4_64.hdf5")
        #print("aff weights")
        classes = 2
    if weights == 'recola':
        weights_path = os.path.join(basepath,"SENET_50_RECOLA_from_RAF.hdf5")
        #print("recola weights")
        classes = 2

    seres50 = SEResNet50(input_shape=(input._keras_shape[2],input._keras_shape[3],input._keras_shape[4]), classes=classes)
    # load FER weights
    

    seres50.load_weights(weights_path)
    # pop the last classification layers
    #seres50.summary()
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


    Rnn_attention = RNNStackedAttention(reshape_dim, cells, return_sequences=True)
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

    model = Model(input, x)
    model.layers[1].trainable = train_all_baseline

    return model

if __name__ == "__main__":
    x = Input((16,224,224,3))
    model =SharmaNet(x)
    #print(model.summary())
