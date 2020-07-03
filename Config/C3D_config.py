from datetime import datetime
import os

TIME_STEP = 16
INPUT_SHAPE = (TIME_STEP,112,112,3)

############ TRAINING PARAMETERS #############
DIR_OUTPUT_TRAIN = "/data/s4180941/AllTrainHistories"

INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_EPOCHS = 60
N_TRAINING_EPOCHS = 240
BATCH_SIZE = 16

CLASSES = ["Angry", "Disgust","Fear","Happy","Neutral","Sad","Surprise"]

#Weights
WEIGHTS_SPORTS1M_PATH = '/data/s4180941/Models/Weights/sports1M_weights_tf.h5'

def get_output_train_path(dir_out_train = DIR_OUTPUT_TRAIN, net_name = 'c3d', name_dataset = 'afew',info_experiments='c3d',lr = INITIAL_LEARNING_RATE,lrdf=LEARNING_RATE_DECAY_FACTOR,lrde=LEARNING_RATE_DECAY_EPOCHS,ne=N_TRAINING_EPOCHS,bs=BATCH_SIZE,ts=TIME_STEP, opt='sgd'):
    today = datetime.now()
    d = today.strftime('%d_%b_%Y_%H_%M')
    out_path = '%s-ts-%s-lr-%s-lrdf-%s-lrde-%s-ep-%s-b-%s-opt-%s'%(d,str(ts),str(lr),str(lrdf),str(lrde),str(ne),str(bs),str(opt))

    directory = os.path.join(dir_out_train,net_name,name_dataset,info_experiments,out_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory