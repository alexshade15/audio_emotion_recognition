import csv
import glob
import os
import pathlib
import random
import subprocess
import sys
import cv2
#import dlib
import joblib
import numpy as np
from tqdm import tqdm
import shutil
# import Models.Config.SEResNet50_config as config
from mtcnn.mtcnn import MTCNN

base_path = os.path.dirname(os.path.abspath(__file__))


###AFEW utils


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty #print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % (cm[i, j])
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def filter_blacklist(data):
    blacklist = ['010924040', '005221760', '010915080', '004510640']

    index_to_remove = []
    for i, item in enumerate(data):
        if item['info']['video_name'] in blacklist:
            index_to_remove.append(i)

    for index in index_to_remove:

        try:
            removed = data[index]
            del data[index]

            print("Removed: " + removed['info']['video_name'])
        except:
            print("Error removing: ", removed['info']['video_name'])

    return data


def _read_dataset(partition, input_path_ds, output_path_cache=base_path + '/CacheFrameProcessing',
                  debug_max_num_samples=None, cache_p=None):
    """read a partition of dataset"""
    #print("Init reading from video files")
    data = []
    if not os.path.isdir(output_path_cache):
        os.makedirs(output_path_cache)
    # iterate partition
    for set in list_dirs(input_path_ds):
        if partition == os.path.basename(set):
            #print("Processing partition: ", partition)
            # for this partition extract all video frames
            for class_dir in tqdm(list_dirs(set)):
                #print("Procssing class: ", os.path.basename(class_dir))
                # init params
                openface_fdir = ""
                label = os.path.basename(class_dir)
                # exctract video frames for any video in a class
                openface_fdir, _ = extract_frames_from_video_folder(class_dir, output_path_cache, debug_max_num_samples,
                                                                    cache_p, partition)
                # preprocess every video frame by detectding and aligning faces
                returned_sequences, map_infos = pre_process_video(openface_fdir, output_path_cache, cache_p, partition)
                # append processed data
                data += process_data(returned_sequences, map_infos, label)
    # check dataset integrity and get statistics
    data = check_data(data, output_path_cache, cache_p, partition, input_path_ds)

    # flush
    shutil.rmtree(output_path_cache)

    return data


def recover_data(input_path_ds, output_cache_path, cache_p, partition, failed_sequences):
    #print("Recovering failed videos")
    recovered = []
    recover_path = output_cache_path
    if not os.path.isdir(recover_path):
        os.makedirs(recover_path)
    # iterate partition
    for set in list_dirs(input_path_ds):
        if partition == os.path.basename(set):
            for class_dir in list_dirs(set):
                file_list = glob.glob('{}/*.avi'.format(class_dir))
                file_list.sort()
                for f in range(0, file_list.__len__()):
                    aviName = file_list[f].split('\\')[(-1)].rstrip('.avi')
                    for item in failed_sequences:
                        if aviName == item[0]:
                            shutil.copy(file_list[f], recover_path)

    openface_fdir, _ = extract_frames_from_video_folder(recover_path, recover_path, None, cache_p, partition)
    # generate all bbox for failed video with our detector
    fd = MTCNN(steps_threshold=[0.35, 0.4, 0.5, 0.6])

    bbox_dir, openface_fdir = get_bbox(recover_path, fd, openface_fdir)

    # preprocess every video frame by using our bbox for aligning faces
    returned_sequences, map_infos = pre_process_video(openface_fdir, output_cache_path, cache_p, partition, bbox_dir)

    extra_map_infos = []
    for i, seq in enumerate(returned_sequences):
        if len(seq) == 0:
            openface_fdir.clear()
            openface_fdir.append(os.path.join(recover_path, map_infos[i]['video_name']))
            extra_seq, extra_map = pre_process_video(openface_fdir, output_cache_path, cache_p, partition,
                                                     bbox=bbox_dir, as_img=True)

            returned_sequences += extra_seq
            extra_map_infos += extra_map

    map_infos += extra_map_infos

    for i, seq in enumerate(returned_sequences):
        new_seq = list()
        new_map_info = list()
        item_to_del = map_infos[i]['video_name']
        label = ""
        if len(seq) > 0:
            for x in failed_sequences:
                if item_to_del == x[0]:
                    label = x[1]

            new_seq.append(returned_sequences[i])
            new_map_info.append(map_infos[i])
            recovered += process_data(new_seq, new_map_info, label)
            #print("Successful recovered: " + item_to_del)
        else:
            print("Error recovering: " + item_to_del + " , Skipping!")

    #print("End recovering failed data")
    return recovered


def get_bbox(recover_path, fd, openface_fdir):
    bb_dir = os.path.join(recover_path + "/bb")
    if not os.path.exists(bb_dir):
        os.makedirs(bb_dir)

    for dir in list_dirs(recover_path):
        if os.path.basename(dir) != "bb":
            deleted = 0
            for file in os.listdir(dir):
                if file.endswith(".png"):

                    frame = cv2.imread(os.path.join(dir, file))
                    faces = fd.detect_faces(frame)
                    if len(faces) != 0:
                        bounding_box = faces[0]['box']
                        keypoints = faces[0]['keypoints']

                        width = bounding_box[2]
                        height = bounding_box[3]
                        txs = bounding_box[0]
                        tys = bounding_box[1]

                        new_width = int(width * 1.0323)
                        new_height = int(height * 0.7751)
                        new_txs = int(width * (-0.0075) + txs)
                        new_tys = int(height * (0.2459) + tys)

                        file_bb = open(bb_dir + "/" + file[:-4] + ".txt", "w")
                        file_bb.write(str(new_txs) + " " + str(new_tys) + " " + str(new_txs + new_width) + " " + str(
                            new_tys + new_height))
                        file_bb.close()
                    else:
                        deleted += 1
                        os.remove(os.path.join(dir, file))

            if len(os.listdir(dir)) == 0:
                for x in openface_fdir:
                    if os.path.basename(dir) in x:
                        openface_fdir.remove(x)
                shutil.rmtree(dir)
                #print("empty deleted: ", str(dir))

    return bb_dir, openface_fdir


def process_data(sequences, infos, label):
    data = []
    for i in range(len(sequences)):
        example = {
            'frames': sequences[i],
            'label': label,
            'info': infos[i],
        }
        data.append(example)
    return data


def check_data(data, output_cache_path, cache_p, partition, input_path_ds):
    """Check data video integrity filtering out bad sequences, in addition a statistics log will be stored"""
    total_frames = 0  # total frames in data
    tatal_frames_discarded = 0  # without face or with wrong prediction
    total_faces_recognized_percentage = list()  # percentage of face recognition/alignment success
    total_failed_sequences = list()  # will contain all video's names failed during pre process
    #print("Checking data integrity")
    # open statistic file in order to store statistics data
    csv.register_dialect('mydialect', delimiter=';', quotechar='"', lineterminator='\r\n', quoting=csv.QUOTE_MINIMAL)
    with open(os.path.join(cache_p, 'dataset_' + partition + '_statistics.csv'), 'w', newline='') as stats_file:
        #print("Stats log file opened")
        writer = csv.writer(stats_file, dialect='mydialect')
        writer.writerow(["Video", "Label", "Total frames", "Discarded frames", "face_presence_percentage"])
        item_to_del = []
        # iterate over all items
        for item in data:
            info = item['info']
            if len(item['frames']) > 0:
                writer.writerow([info['video_name'], item['label'], info['total_frames'], info['discarded_frames'],
                                 info['face_present_percentage']])
                # update global stats variable
                total_frames += info['total_frames']
                tatal_frames_discarded += info['discarded_frames']
                total_faces_recognized_percentage.append(info['face_present_percentage'])
            else:
                total_failed_sequences.append((info['video_name'], item['label']))
                item_to_del.append(item)

        for item in item_to_del: data.remove(item)

        # recover failed_sequences if there are
        if len(total_failed_sequences) > 0:
            # write dataset stats
            writer.writerow([' '])
            writer.writerow(['Recovered failed videos during acquisition'])
            writer.writerow(["Video", "Label", "Total frames", "Discarded frames", "face_presence_percentage"])
            recovered = recover_data(input_path_ds, output_cache_path, cache_p, partition, total_failed_sequences)
            total_failed_sequences.clear()
            item_to_del.clear()
            # update new statistics based on new recovered videos
            for item in recovered:
                info = item['info']
                if len(item['frames']) > 0:
                    writer.writerow([info['video_name'], item['label'], info['total_frames'], info['discarded_frames'],
                                     info['face_present_percentage']])
                    # update global stats variable
                    total_frames += info['total_frames']
                    tatal_frames_discarded += info['discarded_frames']
                    total_faces_recognized_percentage.append(info['face_present_percentage'])
                else:
                    total_failed_sequences.append((info['video_name'], item['label']))
                    item_to_del.append(item)
            for item in item_to_del: recovered.remove(item)

            data += recovered

        # write dataset stats

        writer.writerow([' '])
        writer.writerow(['Dataset statistics'])
        writer.writerow(["Total frames", "Total discarded frames", "face_presence_percentage_mean", "Failed sequences"])
        writer.writerow([total_frames, tatal_frames_discarded, np.mean(total_faces_recognized_percentage),
                         '\r\n'.join([x[0] for x in total_failed_sequences])])
        stats_file.close()
    #print("End check data integrity")

    return data


def list_dirs(directory):
    """Returns all directories in a given directory"""
    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]



def extract_frames_from_video_folder(input_avi, output_path_cache, debug_max_num_samples, cache_p, partition, read_all = False, video_format = '.avi', existent_file_list = None):
    """Extract frames from a folder(class)"""
    if existent_file_list is None:
        if not read_all:
            file_list = glob.glob(('{}/*'+video_format).format(input_avi))
        else:
            file_list = glob.glob(input_avi+"/*")
    else:
        file_list = existent_file_list

    file_list.sort()
    data = []
    error_video = []
    # iterate over all video in dir
    openface_fdir = []

    #print("Init Frames Extraction")
    current_num_samples = 0

    #print("Extracting")
    for f in tqdm(range(0, file_list.__len__())):
        file_to_read = os.path.normpath(file_list[f])
        try:
            aviName = file_to_read.split('/')[(-1)].replace('.avi','')
            aviName = aviName.replace('.mp4','')

            # get path and file name
            save_path = '{}/{}'.format(output_path_cache, aviName)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            output = '{}/{}-%6d_frame.png'.format(save_path, aviName)
            # get aspect ratio
            asr = get_output_size(file_list[f])
            # extract all frames from a video
            extract_frames(file_list[f], output, asr, cache_p, partition)
            openface_fdir.append(save_path)
            if debug_max_num_samples is not None:
                if current_num_samples == debug_max_num_samples - 1:
                    break
        except:
            # check and count video lost
            error_video.append(aviName)
            #print(aviName + ' ffmpeg failed' + '\n')

        current_num_samples += 1

    #print("End Frames Extraction")

    return openface_fdir, error_video


def extract_frames(src, dest, asr, cache_p, partition):
    """Call ffmpeg service and save all frames in dest folder"""
    #print("Calling FFMPEG on video: ", os.path.basename(src))

    # command = ["ffmpeg", "-i", src,"-s", asr, "-q:a", "1", dest]
    command = ['ffmpeg', '-loglevel', 'info', '-hide_banner', '-nostats', '-i', src, '-s', asr, '-q:a', '1', dest]

    try:
        log_file = open(os.path.join(cache_p, 'FFMPEG_output_' + partition + '.log'), "a")
        p = subprocess.Popen(command, stdout=log_file, stderr=log_file).wait()
        log_file.close()
    except Exception as e:
        print(e)


def refactor_output_sequence(frames_dir, avi_name):
    new_folder_align = os.path.join(frames_dir, (avi_name + "_aligned"))
    # creating dir for dataset cache

    if not os.path.isdir(new_folder_align):
        os.makedirs(new_folder_align)
    with open(os.path.join(frames_dir, avi_name + '.csv'), 'w', newline='') as stats_file:

        writer = csv.writer(stats_file)
        writer.writerow(["frame", " face_id", " timestamp", " confidence", " success"])
        for file in (list_dirs(frames_dir)):
            if len(str(os.path.basename(file))) > len(avi_name) and avi_name in os.path.basename(
                    file) and file.is_dir() and str(os.path.basename(file)) != str(os.path.basename(new_folder_align)):
                frame_number = int(os.path.basename(file)[len(avi_name) + 1:].replace("_frame_aligned", ""))
                to_delete_path = str(file).replace("_aligned", ".csv")
                new_face_file_name = 'frame_det_00_{:06d}.png'.format(frame_number)
                new_file_path = os.path.join(new_folder_align, new_face_file_name)
                file = str(file) + "/face_det_000000.png"
                if os.path.exists(file):
                    os.rename(file, new_file_path)
                    writer.writerow([str(frame_number), " 0", " 0.000", " 1", " 1"])
                else:
                    writer.writerow([str(frame_number), " 0", " 0.000", " 0", " 0"])
                os.remove(to_delete_path)
    stats_file.close()

def openface_call(openface_fdir, out_dir, cache_p, partition, bbox = None, as_img=True):
    """preprocess video"""

    # create command for open face
    if as_img:
        command = ['/data/s4179447/OpenFace/build/bin/FaceLandmarkImg']

    else:
        command = ['/data/s4179447/OpenFace/build/bin/FeatureExtraction']

    for _dir in openface_fdir:
        command.append("-fdir")
        command.append(_dir)

    if bbox is not None:
        command.append("-bboxdir")
        command.append(bbox)

    resize_shape = 400
    scale = 1.46
    # resize_shape = 200
    # scale = 0.73

    command += ['-out_dir', out_dir, '-simsize', str(resize_shape), '-simscale', str(scale),
                '-format_aligned', 'jpg', '-nomask', '-simalign', '-wild', '-multi_view', '1','-nobadaligned']

    try:
        #print("Calling OpenFace")
        log_file = open(os.path.join(cache_p, 'OpenFace_output_' + partition + '.log'), "a")
        p = subprocess.Popen(command, stdout=log_file, stderr=log_file).wait()
        log_file.close()

        #print("End OpenFace")


    except Exception as e:
        print(e)
        sys.exit(-1)




def pre_process_video(openface_fdir, frames_dir, cache_p, partition, resize_shape=(224, 224), bbox=None, as_img=False):
    aligned_videos = []
    all_maps = []
    #print("Init pre processing")

    openface_call(openface_fdir,frames_dir,cache_p,partition,bbox,as_img)
    # theshold for diltering out bad faces
    threshold_detection = 0.1

    if as_img:
        refactor_output_sequence(frames_dir, str(os.path.basename(openface_fdir[0])))

    # keep needed info from openface csv out
    for filename in os.listdir(frames_dir):

        if filename.endswith(".csv"):
            aligned_frames = []
            filename = filename[:-4]
            aligned_frames_dir = frames_dir + "/" + filename + "_aligned"
            # open csv
            with open(frames_dir + "/" + filename + ".csv", mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',')
                line_count = 0
                map_info = {}
                map_frame = {}
                map_info['video_name'] = filename
                readed_frames = 0
                discarded_frames = 0
                for row in csv_reader:
                    if int(row[' success']) == 1 and float(row[' confidence']) > threshold_detection:
                        aligned_frame = '{}/frame_det_00_{:06d}.png'.format(aligned_frames_dir, int(row['frame']))
                        aligned_frames.append(cv2.imread(aligned_frame))
                        map_frame[row['frame']] = row[' confidence']
                    else:
                        discarded_frames += 1
                    readed_frames = int(row['frame'])

                csv_file.close()
                map_info['total_frames'] = readed_frames
                map_info['discarded_frames'] = discarded_frames
                map_info['face_present_percentage'] = np.round((readed_frames - discarded_frames) / readed_frames, 2)
                map_info['detections_info'] = map_frame
                all_maps.append(map_info)
                aligned_videos.append(aligned_frames)

                if len(aligned_frames) > 0:
                    shutil.rmtree(frames_dir + "/" + filename)
                # when everything is done flush directories
                shutil.rmtree(frames_dir + "/" + filename + "_aligned")
                os.remove(frames_dir + "/" + filename + ".csv")
    #print("End pre processing")

    return aligned_videos, all_maps


def get_output_size(path, fixed=True, w=720, h=480):
    """given input path of video, returns it's width and height"""
    cap = cv2.VideoCapture(path)
    if fixed:
        width = w
        height = h
    else:
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return '{}x{}'.format(width, height)


def split_video(item=None, split_len=16, partition='Train'):
    splitted_video = []
    video = item['frames']
    label = item['label']
    len_video = len(video)
    steps = len_video // split_len
    rest = len_video % split_len
    i = 0
    # if video len is > of split len
    if steps > 0:
        # get all possible sequences
        while i < steps:
            start = i * split_len
            stop = (i * split_len) + split_len
            actual = np.array(video[start:stop])
            item = {
                'frames': actual,
                'label': label,
            }
            splitted_video.append(item)
            i += 1
        pads = []
        # do padding if there are enough samples left
        if 'val' not in partition.lower():
            #print('Padding on train gen video')
            if rest >= (split_len / 2):
                for i in range(split_len - rest):
                    pads.append(video[-1])
                start = stop
                last = np.concatenate((video[start:], pads), axis=0)
                item = {
                    'frames': np.array(last),
                    'label': label,
                }
                splitted_video.append(item)
    # do padding il video_len is < split_len
    elif steps == 0:
        rest = split_len - len_video
        pads = []
        for i in range(rest):
            pads.append(video[-1])
            last = np.concatenate((video, pads), axis=0)
        item = {
            'frames': np.array(last),
            'label': label,
        }
        splitted_video.append(item)
    return splitted_video


def top_left(f):
    return (f['roi'][0], f['roi'][1])


def bottom_right(f):
    return (f['roi'][0] + f['roi'][2], f['roi'][1] + f['roi'][3])


def enclosing_square(rect):
    def _to_wh(s, l, ss, ll, width_is_long):
        if width_is_long:
            return l, s, ll, ss
        else:
            return s, l, ss, ll

    def _to_long_short(rect):
        x, y, w, h = rect
        if w > h:
            l, s, ll, ss = x, y, w, h
            width_is_long = True
        else:
            s, l, ss, ll = x, y, w, h
            width_is_long = False
        return s, l, ss, ll, width_is_long

    s, l, ss, ll, width_is_long = _to_long_short(rect)

    hdiff = (ll - ss) // 2
    s -= hdiff
    ss = ll

    return _to_wh(s, l, ss, ll, width_is_long)


def add_margin(roi, qty):
    return (
        (roi[0] - qty),
        (roi[1] - qty),
        (roi[2] + 2 * qty),
        (roi[3] + 2 * qty))


def cut(frame, roi):
    pA = (int(roi[0]), int(roi[1]))
    pB = (int(roi[0] + roi[2] - 1), int(roi[1] + roi[3] - 1))  # pB will be an internal point
    W, H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0] >= 0 else 0
    A1 = pA[1] if pA[1] >= 0 else 0
    data = frame[A1:pB[1], A0:pB[0]]
    if pB[0] < W and pB[1] < H and pA[0] >= 0 and pA[1] >= 0:
        return data
    w, h = int(roi[2]), int(roi[3])
    img = np.zeros((h, w, frame.shape[2]), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0] < 0 else 0
    offY = int(-roi[1]) if roi[1] < 0 else 0
    np.copyto(img[offY:offY + data.shape[0], offX:offX + data.shape[1]], data)
    return img


def cut_centered(frame, shape=(224, 224), random=True, random_values=None, max_change_fraction=0.045,
                 only_narrow=False):
    from PIL import Image
    left = int((frame.shape[1] - shape[0]) / 2)
    top = int((frame.shape[1] - shape[0]) / 2)
    right = int((frame.shape[1] + shape[0]) / 2)
    bottom = int((frame.shape[1] + shape[0]) / 2)
    if random:
        if random_values is None:
            sigma = shape[0] * max_change_fraction
            xy = _random_normal_crop(2, sigma, mean=-sigma / 5).astype(int)
            wh = _random_normal_crop(2, sigma * 2, mean=sigma / 2, positive=only_narrow).astype(int)
        else:
            xy, wh = random_values
    else:
        xy = [0, 0]
        wh = [0, 0]

    return frame[(top + wh[0]):(bottom + wh[0]), (left + xy[0]):(right + xy[0]), :]


def pad(img):
    w, h, c = img.shape
    if w == h:
        return img
    size = max(w, h)
    out = np.zeros((size, size, c))
    np.copyto(out[0:w, 0:h], img)
    return out


def findRelevantFace(objs, W, H):
    mindistcenter = None
    minobj = None
    for o in objs:
        cx = o['roi'][0] + (o['roi'][2] / 2)
        cy = o['roi'][1] + (o['roi'][3] / 2)
        distcenter = (cx - (W / 2)) ** 2 + (cy - (H / 2)) ** 2
        if mindistcenter is None or distcenter < mindistcenter:
            mindistcenter = distcenter
            minobj = o
    return minobj


tmp_A = []
FIT_PLANE_SIZ = 16
for y in np.linspace(0, 1, FIT_PLANE_SIZ):
    for x in np.linspace(0, 1, FIT_PLANE_SIZ):
        tmp_A.append([y, x, 1])
Amatrix = np.matrix(tmp_A)


def _fit_plane(im):
    original_shape = im.shape
    if len(im.shape) > 2 and im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (FIT_PLANE_SIZ, FIT_PLANE_SIZ))
    if im.dtype == np.uint8:
        im = im.astype(float)
    # do fit
    A = Amatrix
    tmp_b = []
    for y in range(FIT_PLANE_SIZ):
        for x in range(FIT_PLANE_SIZ):
            tmp_b.append(im[y, x])
    b = np.matrix(tmp_b).T
    fit = (A.T * A).I * A.T * b

    fit[0] /= original_shape[0]
    fit[1] /= original_shape[1]

    def LR(x, y):
        return np.repeat(fit[0] * x, len(y), axis=0).T + np.repeat(fit[1] * y, len(x), axis=0) + fit[2]

    xaxis = np.array(range(original_shape[1]))
    yaxis = np.array(range(original_shape[0]))
    imest = LR(yaxis, xaxis)
    return np.array(imest)


def linear_balance_illumination(im):
    if im.dtype == np.uint8:
        im = im.astype(float)
    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)
    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    imout = im.copy()
    imest = _fit_plane(im[:, :, 0])
    imout[:, :, 0] = im[:, :, 0] - imest + np.mean(imest)
    if im.shape[2] > 1:
        imout = cv2.cvtColor(imout, cv2.COLOR_YUV2BGR)
    return imout.reshape(im.shape)


def mean_std_normalize(inp):
    std = inp.flatten().std()
    if std < 0.001:
        std = 0.001
    return (inp - inp.flatten().mean()) / inp.flatten().std()


def _random_normal_crop(n, maxval, positive=False, mean=0):
    gauss = np.random.normal(mean, maxval / 2, (n, 1)).reshape((n,))
    gauss = np.clip(gauss, mean - maxval, mean + maxval)
    if positive:
        return np.abs(gauss)
    else:
        return gauss


def random_change_image(img, random_values=(
        _random_normal_crop(1, 0.5, mean=1)[0], _random_normal_crop(1, 48)[0], random.randint(0, 1))):
    # brightness and contrast
    a, b, random = random_values
    img = (img - 128.0) * a + 128.0 + b
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # flip
    if random:
        img = np.fliplr(img)
    return img


def random_change_roi(roi, max_change_fraction=0.045, only_narrow=False, random_values=None):
    # random crop con prob + alta su 0 (gaussiana)
    sigma = roi[3] * max_change_fraction
    if random_values is None:
        xy = _random_normal_crop(2, sigma, mean=-sigma / 5).astype(int)
        wh = _random_normal_crop(2, sigma * 2, mean=sigma / 2, positive=only_narrow).astype(int)
    else:
        xy, wh = random_values
    #print("orig roi: %s" % str(roi))
    #print("rand changes -> xy:%s, wh:%s" % (str(xy), str(wh)))
    roi2 = (roi[0] + xy[0], roi[1] + xy[1], roi[2] - wh[0], roi[3] - wh[1])
    #print("new roi: %s" % str(roi2))

    return roi2


def roi_center(roi):
    return (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2)


def random_image_rotate(img, rotation_center, random_angle_deg=_random_normal_crop(1, 10)[0]):
    angle_deg = random_angle_deg
    M = cv2.getRotationMatrix2D(rotation_center, angle_deg, 1.0)
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2])
    return nimg.reshape(img.shape)


def random_image_skew(img, rotation_center, random_skew=_random_normal_crop(2, 0.1, positive=True)):
    s = random_skew
    M = np.array([[1, s[0], 1], [s[1], 1, 1]])
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2])
    return nimg.reshape(img.shape)


def equalize_hist(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(img)


def draw_emotion(y, w, h):
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    COLORS = [(120, 120, 120), (50, 50, 255), (0, 255, 255), (255, 0, 0), (0, 0, 140), (0, 200, 0), (42, 42, 165),
              (100, 100, 200), (170, 170, 170), (80, 80, 80)]
    emotionim = np.zeros((w, h, 3), dtype=np.uint8)
    barh = h // len(EMOTIONS)
    MAXEMO = np.sum(y)
    for i, yi in enumerate(y):
        # #print((EMOTIONS[i], yi))
        p1, p2 = (0, i * barh), (int(yi * w // MAXEMO), (i + 1) * 20)
        # cv2.rectangle(emotionim, p1,p2, COLORS[i], cv2.FILLED)
        cv2.putText(emotionim, "%s: %.1f" % (EMOTIONS[i], yi), (0, i * 20 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255))
    return emotionim


def show_frame(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 20)
    fontScale = 0.3
    fontColor = (255, 255, 255)
    lineType = 1
    cv2.putText(frame,
                text,
                position,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
