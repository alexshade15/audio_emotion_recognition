import csv
import os
from glob import glob
import pickle
import shutil
import sys
from pathlib import Path
import warnings
import random

sys.path.append("/user/rpalladino/")
from Dataset.Dataset_Utils.facedetect_vggface2.MTCNN_detector import MTCNN_detector
from Dataset.Dataset_Utils.datagen import DataGenerator
from Dataset.Dataset_Utils.generate_detections import create_detections_map
from Dataset.Dataset_Utils.dataset_recovery import recover_dataset
from Dataset.Dataset_Utils.double_faces_extraction import double_faces_extraction
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, TimeDistributed
from keras.models import Model

warnings.filterwarnings('ignore', category=FutureWarning)
import cv2
from Dataset.Dataset_Utils.augmenter import DefaultAugmenter, NoAug
from tqdm import tqdm
import joblib
from Dataset.Dataset_Utils.dataset_tools import openface_call, extract_frames, get_output_size

#print("start exec")
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
cache_p = '/user/rpalladino/Dataset/AFEW/AFEW_Cache'
input_p_ds = '/user/rpalladino/Dataset/AFEW/'
videos_path = '/user/rpalladino/Dataset/AFEW/videos'


class AFEW_dataset:
    gen = None
    partition = None

    def __init__(self, partition='Train', input_path=input_p_ds, videos_path=videos_path,
                 cache_path=cache_p):
        self.info = list()
        self.partition = partition
        #print("Loading: ", partition)

        # first stage detections
        cache_path_partition = os.path.join(cache_path, partition)
        Path(cache_path_partition).mkdir(parents=True, exist_ok=True)
        cache_file_name_openface_done = '%s.%s.openface' % ("afew", partition)
        self.aligned_dir = os.path.join(input_path, "aligned", partition)
        Path(self.aligned_dir).mkdir(parents=True, exist_ok=True)
        self.moved_videos_invalid = os.path.join(input_path, "invalid_moved_videos", partition)
        Path(self.moved_videos_invalid).mkdir(parents=True, exist_ok=True)

        detections_dir = os.path.join(input_path, "detections", partition)
        Path(detections_dir).mkdir(parents=True, exist_ok=True)
        recover_dir = os.path.join(input_path, "recover", partition)
        Path(recover_dir).mkdir(parents=True, exist_ok=True)
        fd = MTCNN_detector(steps_threshold=[0.3, 0.4, 0.5])
        video_path_partition = os.path.join(videos_path, partition)

        # openface preprocessing
        try:
            with open(os.path.join(cache_path_partition, cache_file_name_openface_done), 'rb') as f:
                print("Openface preprocess already done")

        except FileNotFoundError:
            #print('File not found,creating...')
            #print('Openface file not found, initializing openface...')

            temp_path_extraction = os.path.join(input_path, "temp_extraction", partition)
            Path(temp_path_extraction).mkdir(parents=True, exist_ok=True)

            for label_folder in glob(os.path.join(video_path_partition, '*')):
                label = os.path.basename(label_folder)
                aligned_dir_label = os.path.join(self.aligned_dir, label)
                Path(aligned_dir_label).mkdir(parents=True, exist_ok=True)

                moved_videos_invalid_label = os.path.join(input_path, "invalid_moved_videos", partition, label)
                Path(moved_videos_invalid_label).mkdir(parents=True, exist_ok=True)
                for video in glob(os.path.join(label_folder, '*')):
                    video_name = os.path.basename(video)[:-4]
                    temp_path_extraction_single_video = os.path.join(temp_path_extraction, video_name)
                    Path(temp_path_extraction_single_video).mkdir(parents=True, exist_ok=True)
                    w, h = get_output_size(video, True)
                    asr = '{}x{}'.format(w, h)
                    output = '{}/{}-%6d_frame.png'.format(temp_path_extraction_single_video, video_name)
                    extract_frames(video, output, asr, cache_p, partition)
                    openface_call([temp_path_extraction_single_video], aligned_dir_label, cache_p, partition,
                                  as_img=False)
                    shutil.rmtree(temp_path_extraction_single_video)

            #print("doing backup on cache file")
            map_dump = {}
            with open(os.path.join(cache_path_partition, cache_file_name_openface_done), 'wb') as f:
                joblib.dump(map_dump, f)

        cache_file_name_recover_single_done = '%s.%s.recover_single' % ("afew", partition)

        try:
            with open(os.path.join(cache_path_partition, cache_file_name_recover_single_done), 'rb') as f:
                recovered_single = pickle.load(
                    open(os.path.join(cache_path_partition, cache_file_name_recover_single_done), 'rb'))

        except FileNotFoundError:
            #print('File recover single face not found,creating...')
            #print("Recovering single face")
            # aligned dir recover is the output folder containing aligned recovered videos

            for label_folder in glob(os.path.join(video_path_partition, '*')):
                label = os.path.basename(label_folder)
                aligned_dir_label = os.path.join(self.aligned_dir, label)
                Path(aligned_dir_label).mkdir(parents=True, exist_ok=True)
                aligned_moved_videos_label = os.path.join(self.moved_videos_invalid, label)
                recovered_single = recover_dataset(aligned_dir_label, aligned_moved_videos_label, label_folder,
                                                   detections_dir, recover_dir, cache_p, fd)
            pickle.dump(recovered_single, open(os.path.join(cache_path_partition, cache_file_name_recover_single_done),
                                               'wb'))

    def get_video_generator(self, target_shape=(224, 224, 3), augmenter=DefaultAugmenter(), preprocessing='full',
                            batch_size=8, n_seq_per_epoch=24, sequence_len=16, split_video_len=16, random_windows=True,
                            max_invalid=8):
        return DataGenerator(self.aligned_dir, '', target_shape=target_shape, batch_size=batch_size,
                             n_seq_per_epoch=n_seq_per_epoch, sequence_len=sequence_len,
                             split_video_len=split_video_len, augmenter=augmenter, preprocessing=preprocessing,
                             random_windows=random_windows, dataset='afew', max_invalid=max_invalid)


# if __name__ == "__main__":
#     dt = AFEW_dataset(partition='Val')
#     dg = dt.get_video_generator(random_windows=True, augmenter=DefaultAugmenter(), n_seq_per_epoch=2400)
#     for batchs_img, batchs_lbl in dg:
#         #print(batchs_img.shape)
#         for sequence, label in zip(batchs_img, batchs_lbl):
#
#             for x in sequence:
#                 MAX = np.amax(x)
#                 MIN = np.amin(x)
#                 x = 255 * (x - MIN) / (MAX - MIN)
#                 x = x.clip(0, 255).astype(np.uint8)
#                 cv2.imshow('x', x)
#                 #print(label)
#                 if 0xff & cv2.waitKey(40) == 27:
#                     sys.exit(0)
#
#             #print("\n\nNEW SEQ")
