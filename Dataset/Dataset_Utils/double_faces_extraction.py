import os
import pickle
from pathlib import Path

import cv2
from keras_vggface import VGGFace
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
from Dataset.Dataset_Utils.dataset_tools import add_margin, enclosing_square, cut


def double_faces_extraction(double_face_video_path, detections, temp_dir_double_extraction):
    list_det_final = []
    frame_counter = 0
    face_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    cv2video = cv2.VideoCapture(double_face_video_path)
    video_double = os.path.basename(double_face_video_path)
    video_name = video_double[:-4]
    video_frames_extraction_folder = os.path.join(temp_dir_double_extraction,video_name)

    # print("Processing video")
    analysis_step = True
    right_face = {'best_img': None, 'features': [], 'avg_feature': None, 'rois':[]}
    left_face = {'best_img': None, 'features': [], 'avg_feature': None, 'rois':[]}

    current_annotation = 'right'
    video_frames_extraction_folder_annotation_left = video_frames_extraction_folder + "_left"

    Path(video_frames_extraction_folder_annotation_left).mkdir(parents=True, exist_ok=True)

    video_frames_extraction_folder_annotation_right = video_frames_extraction_folder + "_right"
    Path(video_frames_extraction_folder_annotation_right).mkdir(parents=True, exist_ok=True)

    counter_for_analysis = 0
    counter_maximum_frame_analysis = 20

    # print("Processing: ", video_name)
    pbar = tqdm(total=len(detections))
    total = len(detections)
    previous_f = {}
    previous_f['roi'] = None
    detections_final = {}
    roi_right = None
    roi_left = None
    while (frame_counter < total):
        # Capture frame-by-frame
        ret, frame = cv2video.read()
        if ret is True:

            if analysis_step and frame_counter % 30 == 0:
                faces = detections[frame_counter]

                if len(faces) == 2:
                    extraction_condition = check_left_right_from_center(faces, frame.shape[1])
                    if extraction_condition:
                        left_face_from_frame, right_face_from_frame = extract_left_right_faces(faces)

                        if current_annotation == 'right':
                            f = right_face_from_frame
                        else:
                            f = left_face_from_frame

                        resized_left_face = cv2.resize(left_face_from_frame['img'], (224, 224))
                        y_left = face_model.predict(np.expand_dims(resized_left_face, axis=0))
                        left_face['features'].append(y_left)

                        resized_right_face = cv2.resize(right_face_from_frame['img'], (224, 224))
                        y_right = face_model.predict(np.expand_dims(resized_right_face, axis=0))
                        right_face['features'].append(y_right)

            elif not analysis_step:
                faces = detections[frame_counter]

                if len(faces) > 1:

                    if current_annotation == 'right':
                        f, pred = findFaceOnSide(face_model, faces, (left_face, right_face), True,
                                                 frame.shape[1], previous_f['roi'])
                    else:
                        f, pred = findFaceOnSide(face_model, faces, (left_face, right_face), False,
                                                 frame.shape[1], previous_f['roi'])



                elif len(faces) == 1:
                    print("only one face")
                    checked_similarity, pred = check_face_similarity(face_model, faces[0],
                                                                     (left_face, right_face))
                    if (checked_similarity == 0 and current_annotation == 'left') or (
                            checked_similarity == 1 and current_annotation == 'right'):
                        f = faces[0]

                    # just a not verified condition
                    ##in this case is a face B
                    else:
                        f = None


                # detection fails --> return map_if_error error
                else:
                    print("detection problem: ", frame_counter)

                    f = None
                if (f is not None) and (f['img'].size != 0):

                    detections_final[frame_counter] = [f]
                    previous_f = f

                    f['roi'] = add_margin(f['roi'], 0.9 * f['roi'][2])
                    f['roi'] = enclosing_square(f['roi'])
                    img = cut(frame, f['roi'])
                    cv2.imwrite(
                        video_frames_extraction_folder + "_" + current_annotation + "/frame-{:06}.png".format(
                            frame_counter), img)


                else:
                    if previous_f['roi'] is None:

                        roi_prov = (100,100,100,100)
                        img_recover = cut(frame, roi_prov)
                        detections_final[frame_counter] = [{'roi':roi_prov}]

                    else:
                        img_recover = cut(frame, previous_f['roi'])
                        detections_final[frame_counter] = [previous_f]

                    cv2.imwrite(
                        video_frames_extraction_folder + "_" + current_annotation + "/frame-{:06}.png".format(
                            frame_counter), img_recover)


        else:
            if not analysis_step:
                print("ret False")

                # in this case just append a None object (interpolation to do after this stage), cv2 video fails

        frame_counter += 1

        if frame_counter == total and analysis_step or counter_for_analysis > counter_maximum_frame_analysis:
            frame_counter = 0
            cv2video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            analysis_step = False
            left_face['avg_feature'] = np.mean(left_face['features'], axis=0)
            right_face['avg_feature'] = np.mean(right_face['features'], axis=0)
            counter_for_analysis = 0
            pbar = tqdm(total=total)

        if frame_counter == total and not analysis_step and current_annotation == 'right':
            cv2video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
            current_annotation = 'left'
            pbar = tqdm(total=total)
            list_det_final.append(detections_final)
            detections_final = {}
            previous_f = {}
            previous_f['roi'] = None

        pbar.update(1)
    pbar.close()

    # When everything done, release the video capture object
    cv2video.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    list_det_final.append(detections_final)

    return list_det_final




def findFaceOnSide(model, faces, faces_left_right, right, width, previous_roi):
    candidates_faces = []
    face_to_compare = None
    offset_roi_tracking = 0.1
    offset_middle_image = 0.1

    if right:
        face_to_compare = faces_left_right[1]
        start_bound = (width / 2) - offset_middle_image * width
    else:
        face_to_compare = faces_left_right[0]
        start_bound = (width / 2) + offset_middle_image * width

    for f in faces:

        if previous_roi is not None:
            x_offset_condition = previous_roi[0] - offset_roi_tracking * previous_roi[0] <= f['roi'][0] <= previous_roi[
                0] + offset_roi_tracking * previous_roi[0]
            y_offset_condition = previous_roi[1] - offset_roi_tracking * previous_roi[1] <= f['roi'][1] <= previous_roi[
                1] + offset_roi_tracking * previous_roi[1]

        else:
            x_offset_condition = False
            y_offset_condition = False

        #in this case is better to choose and stop the analysis?
        if x_offset_condition and y_offset_condition:
            print("buonding box tracking condition verified")
            #resized_face_compare = cv2.resize(f['img'], (224, 224))

            # y_possible_face = model.predict(expand_dims(resized_face_compare, axis=0))
            #
            if right:
                y_compare = faces_left_right[1]['avg_feature']
            else:
                y_compare = faces_left_right[0]['avg_feature']
            #
            # score = cosine(y_compare, y_possible_face)
            # # print("tracking: condizione verificata con score:", score)
            # if score < 0.6:
            #     return f, y_possible_face

            return f, y_compare
        else:
            # print("tracking: condizione NON verificata")

            if right:
                if f['roi'][0] > start_bound:
                    candidates_faces.append(f)


            else:
                face_to_compare = faces_left_right[0]
                if f['roi'][0] < start_bound:
                    candidates_faces.append(f)
    checked_face, prediction = compare_face_candidates(model, face_to_compare, candidates_faces)
    return checked_face, prediction


def check_left_right_from_center(faces, width):
    thresh = 0.93
    if faces[0]['roi'][0] < (width / 2) and faces[1]['roi'][0] > (width / 2):
        if faces[0]['confidence'] >= thresh and faces[1]['confidence'] >= thresh:
            return True
        else:
            return False

    elif faces[1]['roi'][0] < (width / 2 + 0.1 * width) and faces[0]['roi'][0] > (width / 2 - 0.1 * width):
        if faces[0]['confidence'] >= thresh and faces[1]['confidence'] >= thresh:
            return True
        else:
            return False
    else:
        return False

def extract_left_right_faces(faces):
    if faces[0]['roi'][0] < faces[1]['roi'][0]:
        left = faces[0]
        right = faces[1]
    else:
        right = faces[0]
        left = faces[1]

    return left, right

def check_face_similarity(model, face_to_compare, faces_left_right):
    resized_face_compare = cv2.resize(face_to_compare['img'], (224, 224))

    y_compare = model.predict(np.expand_dims(resized_face_compare, axis=0))

    y_left = faces_left_right[0]['avg_feature']

    y_right = faces_left_right[1]['avg_feature']

    score_right = cosine(y_compare, y_right)
    score_left = cosine(y_compare, y_left)

    if score_left > 0.4 and score_right > 0.4:
        return None, None

    if score_right < score_left:
        return 1, y_compare
    else:
        return 0, y_compare


def from_detections_to_faces(frame, faces_info):
    faces = []
    for i in range(len(faces_info['rois'])):
        faces.append(recover_all_face_info_from_index(frame, faces_info, i))
    return faces

def recover_all_face_info_from_index(image, faces_info, index):
    roi = faces_info['rois'][index]
    confidence = faces_info['confidences'][index]
    try:
        landmarks = faces_info['landmarks'][index]
    except:
        landmarks = None
    img = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    f = {}
    f['roi'] = roi
    f['confidence'] = confidence
    f['img'] = img
    f['landmarks'] = landmarks
    return f


def extract_left_right_faces(faces):
    if faces[0]['roi'][0] < faces[1]['roi'][0]:
        left = faces[0]
        right = faces[1]
    else:
        right = faces[0]
        left = faces[1]

    return left, right

def compare_face_candidates(model, face_to_compare, faces, thresh=0.4):
    y_compare = face_to_compare['avg_feature']
    score_to_compare = 99999999
    face_to_return = None
    y_to_return = None
    for face in faces:
        resized_face = cv2.resize(face['img'], (224, 224))
        y_hat = model.predict(np.expand_dims(resized_face, axis=0))
        score = cosine(y_compare, y_hat)

        # todo check

        if score < 0.25:
            # print("condizione di score OK < 0.25")
            return face, y_hat

        if score < score_to_compare and score < 0.4:
            score_to_compare = score
            face_to_return = face
            y_to_return = y_hat

    return face_to_return, y_to_return

