import csv
import os
import pickle
import shutil
from glob import glob
from pathlib import Path
import cv2


from Dataset.Dataset_Utils.dataset_tools import get_output_size, extract_frames, findRelevantFace, openface_call
from Dataset.Dataset_Utils.generate_detections import create_detections_map

def list_dirs(directory):
    """Returns all directories in a given directory"""
    return [f for f in Path(directory).iterdir() if f.is_dir()]

def _recover_video(aligned_dataset_dir, recover_dir_frames, recover_dir_openface, detections, cache_p,
                   video_name_original, shape):
    frame_name = video_name_original + '-{:06d}_frame.png'
    bb_dir = os.path.join(recover_dir_frames + "/bb")
    Path(bb_dir).mkdir(parents=True, exist_ok=True)
    _write_bbox(bb_dir, detections, shape, frame_name, recover_dir_frames, recover_dir_openface)
    openface_call([recover_dir_frames], recover_dir_openface, cache_p, 'recover', bb_dir, True)
    discarded_new = create_aligned_sequence(recover_dir_openface, video_name_original)
    return discarded_new

def recover_dataset(aligned_dataset_dir, moved_invalid_dir, videos_dir, detections_dir, recover_dir, cache_p, fd,
                    minimum_percentage=0.3, single_face=True):
    recovered_list = []
    for file_csv in glob(os.path.join(aligned_dataset_dir, '*.csv')):
        if (single_face and '_right' not in file_csv and '_left' not in file_csv) or (
                (not single_face) and ('_right' in file_csv or '_left' in file_csv)):
            discarded_frames = 0
            readed_frames = 0
            align_info = open(file_csv)
            for i_ali in align_info:
                i_ali = [x.strip() for x in i_ali.split(',')]
                if not i_ali[0].isdigit():
                    continue
                success = bool(int(i_ali[4]))
                if success is False:
                    discarded_frames += 1
                readed_frames += 1
            align_info.close()
            percentage_discarded = discarded_frames / readed_frames
            video_name_orginal = os.path.basename(file_csv[:-4])
            print(video_name_orginal + " discarded: " + str(percentage_discarded))

            if percentage_discarded > minimum_percentage:

                recovered_list.append(video_name_orginal)
                video_name = video_name_orginal.replace('_left', '')
                video_name = video_name.replace('_right', '')
                video_path = os.path.join(videos_dir, video_name)
                if os.path.isfile(video_path + ".mp4"):
                    video_path = video_path + ".mp4"
                elif os.path.isfile(video_path + ".avi"):
                    video_path = video_path + ".avi"
                else:
                    raise
                recover_dir_video = os.path.join(recover_dir, video_name_orginal)

                if single_face:
                    detection_dir_file = os.path.join(detections_dir, video_name, video_name + '.detections')
                else:
                    detection_dir_file = os.path.join(detections_dir, video_name,
                                                      video_name_orginal + '.relevant_detections')

                Path(recover_dir_video).mkdir(parents=True, exist_ok=True)
                w, h = get_output_size(video_path, False)
                asr = '{}x{}'.format(w, h)
                shape = [w, h]

                recover_dir_frames = os.path.join(recover_dir_video, 'frames_extraction')
                Path(recover_dir_frames).mkdir(parents=True, exist_ok=True)
                video_name = os.path.basename(video_path)[:-4]
                output = '{}/{}-%6d_frame.png'.format(recover_dir_frames, video_name_orginal)
                extract_frames(video_path, output, asr, cache_p, 'recover')
                if os.path.isfile(detection_dir_file):
                    detections_map = pickle.load(open(detection_dir_file, "rb"))
                else:
                    Path(os.path.join(detections_dir, video_name)).mkdir(parents=True, exist_ok=True)
                    detections_map = create_detections_map(recover_dir_frames, fd)
                    pickle.dump(detections_map, open(detection_dir_file, "wb"))
                recover_dir_openface = os.path.join(recover_dir_video, 'openface')
                Path(recover_dir_openface).mkdir(parents=True, exist_ok=True)
                # move to another folder in order to store the original invalid video frames

                new_discarded = _recover_video(aligned_dataset_dir, recover_dir_frames, recover_dir_openface, detections_map, cache_p,
                               video_name_orginal, shape)
                print("New discarded percentage: ",new_discarded)
                if new_discarded < percentage_discarded:
                    video_to_move_frames = os.path.join(
                        os.path.join(aligned_dataset_dir, video_name_orginal + '_aligned'))
                    shutil.move(video_to_move_frames, moved_invalid_dir)

                    video_to_move_csv = os.path.join(os.path.join(aligned_dataset_dir, video_name_orginal + '.csv'))
                    shutil.move(video_to_move_csv, moved_invalid_dir)
                    aligned_folder = os.path.join(recover_dir_openface, video_name_orginal + "_aligned")
                    shutil.move(aligned_folder, aligned_dataset_dir)
                    shutil.move(os.path.join(recover_dir_openface, video_name_orginal + '.csv'), aligned_dataset_dir)
                else:
                    print("video "+video_name_orginal+" NOT substituded with a recovered version")
                shutil.rmtree(recover_dir_video)
            else:
                print("video {0} is good with {1} of invalid frames".format(video_name_orginal, percentage_discarded))

    return recovered_list


def _write_bbox(bb_dir, detections, shape, frame_name_format, recover_dir_frames,openface_dir):
    previous_roi = None
    W, H = shape[0], shape[1]

    for k, faces in detections.items():
        if len(faces) != 0:
            f = findRelevantFace(faces, W, H)
            bounding_box = f['roi']
            width = bounding_box[2]
            height = bounding_box[3]
            txs = bounding_box[0]
            tys = bounding_box[1]

            new_width = int(width * 1.0323)
            new_height = int(height * 0.7751)
            new_txs = int(width * (-0.0075) + txs)
            new_tys = int(height * (0.2459) + tys)

            file_name_create = frame_name_format.format(k + 1)[:-4]
            file_bb = open(bb_dir + "/" + file_name_create + ".txt", "w")
            file_bb.write(str(new_txs) + " " + str(new_tys) + " " + str(new_txs + new_width) + " " + str(
                new_tys + new_height))
            file_bb.close()
        else:
            os.remove(os.path.join(recover_dir_frames, frame_name_format.format(k + 1)))
            frame_aligned_folder_empty = os.path.join(openface_dir,frame_name_format.format(k + 1)[:-4]+'_aligned')
            Path(frame_aligned_folder_empty).mkdir(parents=True, exist_ok=True)


def create_aligned_sequence(frames_dir, avi_name):
    new_folder_align = os.path.join(frames_dir, (avi_name + "_aligned"))
    # creating dir for dataset cache
    if not os.path.isdir(new_folder_align):
        os.makedirs(new_folder_align)
    with open(os.path.join(frames_dir, avi_name + '.csv'), 'w', newline='') as stats_file:

        writer = csv.writer(stats_file)
        writer.writerow(["frame", " face_id", " timestamp", " confidence", " success"])
        discarded = 0
        counter_frames = 0
        for file in sorted(list_dirs(frames_dir)):
            if len(str(os.path.basename(file))) > len(avi_name) and avi_name in os.path.basename(
                    file) and file.is_dir() and str(os.path.basename(file)) != str(os.path.basename(new_folder_align)):
                frame_number = int(os.path.basename(file)[len(avi_name) + 1:].replace("_frame_aligned", ""))
                counter_frames += 1
                to_delete_path = str(file).replace("_aligned", ".csv")
                new_face_file_name = 'frame_det_00_{:06d}.jpg'.format(frame_number)
                new_file_path = os.path.join(new_folder_align, new_face_file_name)
                folder_file = file
                file = str(file) + "/face_det_000000.jpg"
                if os.path.exists(file):
                    os.rename(file, new_file_path)
                    writer.writerow([str(frame_number), " 0", " 0.000", " 1", " 1"])
                else:
                    writer.writerow([str(frame_number), " 0", " 0.000", " 0", " 0"])
                    discarded += 1

                if os.path.isfile(to_delete_path):
                    os.remove(to_delete_path)
                shutil.rmtree(folder_file)
    stats_file.close()
    return discarded/counter_frames

