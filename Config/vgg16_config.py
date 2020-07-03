# import the necessary packages
import os

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "/mnt/nvme0n1p1/gallegretti/AFEW_Cropped"

#output directory where to store weights,history etc
OUTPUT_PATH = "/mnt/nvme0n1p1/gallegretti/models/output_training"

# define paths of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "val"

# initialize the list of class label names
CLASSES = ["Angry", "Disgust","Fear","Happy","Neutral","Sad","Surprise"]

# set the batch size when fine-tuning
BATCH_SIZE = 32

# number of epochs
EPOCHS = 2
LAST_EPOCHS = 1

#output directory where to store weights,history etc
CHECKPOINTS_PATH = os.path.sep.join([OUTPUT_PATH, "Checkpoint"])
LOG_TENSORBOARD = os.path.sep.join([OUTPUT_PATH, "Log_TensorBoard"])

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "warmup.png"])