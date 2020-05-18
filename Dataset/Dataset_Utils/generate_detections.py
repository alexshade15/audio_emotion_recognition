import os
from glob import glob
import cv2
from tqdm import tqdm
import sys
sys.path.append('/data/s4179447/')
from Dataset.Dataset_Utils.facedetect_vggface2.MTCNN_detector import MTCNN_detector
from pathlib import Path
from Dataset.Dataset_Utils.dataset_tools import get_output_size, extract_frames

def create_detections_map(video_path, fd):
    frame_counter = 0
    detections = {}
    #print("Detections in progress...\n\n")
    pbar = tqdm()
    for frame_path in sorted(glob(os.path.join(video_path,'*'))):
        # Capture frame-by-frame
        frame = cv2.imread(frame_path)
        faces_final = fd.detect(frame)
        detections[frame_counter] = faces_final
        pbar.update(1)
        frame_counter += 1
    pbar.close()
    return detections


if __name__ == '__main__':
    video_path = '/data/s4179447/Dataset/AFF-Wild/VA/videos/Validation_double/video74.mp4'
    video_name = os.path.basename(video_path)[:-4]
    temp_path_extraction = '/data/s4179447/temp_detection'

    temp_path_extraction_single_video = os.path.join(temp_path_extraction,video_name)
    Path(temp_path_extraction_single_video).mkdir(parents=True, exist_ok=True)
    w,h = get_output_size(video_path,True)
    asr = '{}x{}'.format(w, h)
    output = '{}/{}-%6d_frame.png'.format(temp_path_extraction_single_video, video_name)
    #print("frame extraction")
    extract_frames(video_path,output,asr,'/data/s4179447/','temp')
    fd = MTCNN_detector(steps_threshold = [0.3, 0.5, 0.6])
    create_detections_map(temp_path_extraction_single_video,fd)
