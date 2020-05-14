import os
import sys
from keras.layers import Input
from Dataset.Dataset_Utils.dataset_tools import print_cm
from Dataset.Dataset_Utils.datagen import DataGenerator
from Dataset.Dataset_Utils.augmenter import NoAug
from Models.model_sharma import SharmaNet
from inference import Inference

log_path = "./logs_for_test/"
model_path = "/user/vlongobardi/checkpoint_best.hdf5"
data_path = "/user/rpalladino/Dataset/AFEW/aligned/Val/Fear/000142325.csv"
# data_path = "/user/rpalladino/Dataset/AFEW/aligned/Val/"
# data_path = "/user/rpalladino/Dataset/AFEW/aligned/Val/Fear/000142325_aligned/"
# csv_path = "/user/rpalladino/Dataset/AFEW/aligned/Val/Fear/000142325.csv"

batch_size = 16
n_seq_per_epoch = 24
time_step = 16
classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

x = Input(shape=(time_step, 224, 224, 3))
model = SharmaNet(x, classification=True, weights='afew')
model.load_weights(model_path)

test_genenerator = DataGenerator(data_path, '', batch_size, n_seq_per_epoch, NoAug(), split_video_len=1, max_invalid=12,
                                 test=True)
inference = Inference(model=model, custom_inference=True)

m = {'stride1': "STRIDE 1", 'overlap': "OVERLAP", '': "UNKNOWN"}

log_file = open(os.path.join(log_path, "reports.log"), "w")
old_stdout = sys.stdout
sys.stdout = log_file

for mod in m:
    print(m[mod])

    res = inference.predict_generator(test_genenerator, mode=mod)
    stats = inference.get_stats(res[0], res[1])

    print("###Results ", m[mod], "###")
    for i, el in enumerate(stats):
        if i < 2:
            print_cm(el, classes)
        elif i == 2:
            print("Accuracy score ", m[mod], ": ", el)
        else:
            print("Report")
            print(el)
        print("\n\n")

sys.stdout = old_stdout
log_file.close()
