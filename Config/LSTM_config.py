from datetime import datetime
import os
############ NET LSTM PARAMETERS #############
#input shape
TIME_STEP = 16
FEATURES_DIM = 2048

#YOU CAN ADD LAYERS DIMENSION FROM 1 TO N
LAYERS_SIZE = (128,128,128)
LAYERS_DROPOUT=(0.5,0.5,0.5)

BIDIRECTIONAL = True
BATCH_NORM = True
MOMENTUM = True

BALANCE_DATA = False

############ TRAINING PARAMETERS #############
DIR_OUTPUT_TRAIN = "/data/s4180941/AllTrainHistories"

INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_EPOCHS = 60
N_TRAINING_EPOCHS = 80
BATCH_SIZE = 16

CLASSES = ["Angry", "Disgust","Fear","Happy","Neutral","Sad","Surprise"]

#best trained weight on seresnet50-lstm  with afew acc 0.4829 loss 1.552
SENET50_LSTM_AFEW_WEIGHTS_PATH = '/data/s4175719/AllTrainHistories/seres_all_trainable/afew_from_fer/08_Dec_2019_18_02-nf-4_N-lr-0.0001-lrdf-0.1-lrde-60-ep-180-b-48/best_checkpoint.hdf5'
SENET50_LSTM_AFEW_BAL_WEIGHTS_PATH = '/data/s4175719/AllTrainHistories/seres_lstm/afew_balanced/20_Nov_2019_12_22-lr-0.01-lrdf-0.1-lrde-60-ep-160-b-14-l-128-d-0.7-bd/checkpoint.39.hdf5'

def get_output_train_path(dir_out_train = DIR_OUTPUT_TRAIN, net_name = 'seres_lstm', name_dataset = 'afew',info_experiments='2_livelli_lstm_new',lr = INITIAL_LEARNING_RATE,lrdf=LEARNING_RATE_DECAY_FACTOR,lrde=LEARNING_RATE_DECAY_EPOCHS,ne=N_TRAINING_EPOCHS,bs=BATCH_SIZE,ts=TIME_STEP, opt='sgd'):
    today = datetime.now()
    d = today.strftime('%d_%b_%Y_%H_%M')
    if LEARNING_RATE_DECAY_EPOCHS != N_TRAINING_EPOCHS:
        out_path = '%s-ts-%s-lr-%s-lrdf-%s-lrde-%s-ep-%s-b-%s-opt-%s'%(d,str(ts),str(lr),str(lrdf),str(lrde),str(ne),str(bs),str(opt))
    else:
        out_path = '%s-ts-%s-lr-%s-lrdf-%s-lrde-%s-ep-%s-b-%s'%(d,str(ts),str(lr),'no','no',str(ne),str(bs))

    out_path += '-l'
    for l in LAYERS_SIZE:
        out_path += '-%s'%(str(l))
    out_path += '-d'  
    for drop in LAYERS_DROPOUT:
        out_path += '-%s'%(str(drop))
    if BIDIRECTIONAL:
        out_path += '-bd'
    if MOMENTUM:
        out_path += '-wm'
    if BALANCE_DATA:name_dataset ='afew_balanced'
    

    directory = os.path.join(dir_out_train,net_name,name_dataset,info_experiments,out_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory