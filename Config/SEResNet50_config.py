#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from datetime import datetime
import os

############ NET SERESNET50 PARAMETERS #############

#size of the lpf filter (1 means no filtering) choices=[1, 3, 5, 7] int
LPF_SIZE = 1
#dataset to use for the training str
DATASET = 'AFEW'
#train or test  str
MODE = 'train'
#Last checkpoint to load
LAST_CHECKPOINT = 164
#epoch to be used for testing, mandatory if mode=test  int
TEST_EPOCHS = None
#input shape
INPUT_SHAPE = (224,224,3)

#number of frames keept from video afew dataset
N_FRAMES = 4
FRAMES_LEN = 16

############ TRAINING PARAMETERS #############
DIR_OUTPUT_TRAIN = "/data/s4175719/AllTrainHistories"


INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_EPOCHS = 60
N_TRAINING_EPOCHS = 180
BATCH_SIZE = 48

MOMENTUM = False

CLASSES = ["Angry", "Disgust","Fear","Happy","Neutral","Sad","Surprise"]
NUM_CLASSES = 7

#best trained weight on ferplus
SENET50_FERPLUS_WEIGHTS_PATH = '/home/s4175719/ProjectThesis/Models/Weights/SENET_50_FERPlus_weights.hdf5'
#best trained weight on seresnet50 with afew acc 0.4462 loss 1.585
SENET50_AFEW_WEIGHTS_PATH = '/data/s4175719/AllTrainHistories/seres_all_trainable/afew_from_fer_new/11_Dec_2019_14_34-nf-4_N-lr-0.0001-lrdf-0.1-lrde-60-ep-180-b-48/best_checkpoint.hdf5'
#weight raf dataset
SENET50_RAF_WEIGHTS_PATH = '/data/s4175719/Net_Weights/raf_wd005_mom_lpf0_ep195.hdf5'

def get_output_train_path(dir_out_train = DIR_OUTPUT_TRAIN, net_name = 'seres_all_trainable', name_dataset = 'afew_from_fer_new',lr = INITIAL_LEARNING_RATE,lrdf=LEARNING_RATE_DECAY_FACTOR,lrde=LEARNING_RATE_DECAY_EPOCHS,ne=N_TRAINING_EPOCHS,bs=BATCH_SIZE):

    today = datetime.now()
    d = today.strftime('%d_%b_%Y_%H_%M')
    out_path = '%s-nf-%s_N-lr-%s-lrdf-%s-lrde-%s-ep-%s-b-%s'%(d,str(N_FRAMES),str(lr),str(lrdf),str(lrde),str(ne),str(bs))
    if MOMENTUM: out_path += '-wm'
    directory = os.path.join(dir_out_train,net_name,name_dataset,out_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory



V1_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'
V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'

VGG16_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGG16_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'


RESNET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_resnet50.h5'
RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'

SENET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_senet50.h5'
SENET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5'

SENET50_FERPLUS_WEIGHTS = 'SENET_50_FERPlus_weights.hdf5'

VGGFACE_DIR = 'models/vggface'