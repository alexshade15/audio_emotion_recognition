import csv
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL.Image import Image

from .dataset_tools import findRelevantFace, enclosing_square, cut
from .facedetect_vggface2.MTCNN_detector import MTCNN_detector
from .facedetect_vggface2.face_aligner import FaceAligner
from .facedetect_vggface2.face_detector import FaceDetector


class PointState():

    def __init__(self, point):
        self._point = point

        self._kalman = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        self._is_predicted = False
        self._delta_time = 0.3
        self._accel_noise_mag = 0.5
        self._set_kalman()

    '''
    def update(self,point):
        self._point = (point[0],point[1])
    '''

    def update(self, point):

        measurement = np.zeros((2, 1), dtype=np.float32)

        if point[0] < 0 or point[1] < 0:
            # // update using prediction
            self._predict()
            measurement[0][0] = self._point[0]
            measurement[1][0] = self._point[1]
            self.is_predicted = True
        else:
            # update using measurement
            measurement[0][0] = point[0]
            measurement[1][0] = point[1]
            self.is_predicted = False

        # update using measurements
        estimated = self._kalman.correct(measurement)

        self._point = (estimated[0], estimated[1])

        self._predict()
        # #print("predicted: ",self._point)

    def get_point(self):
        return self._point

    def is_predicted(self):
        return self.is_predicted

    def _predict(self):
        prediction = self._kalman.predict()
        self._point = (prediction[0][0], prediction[1][0])
        return self._point

    def _set_kalman(self):
        self._kalman.transitionMatrix = np.float32(
            [[1, 0, self._delta_time, 0], [0, 1, 0, self._delta_time], [0, 0, 1, 0], [0, 0, 0, 1]])
        self._kalman.statePre = np.zeros((4, 1), dtype=np.float32)

        self._kalman.statePre[0][0] = self._point[0]  # x
        self._kalman.statePre[1][0] = self._point[1]  # y

        self._kalman.statePre[2][0] = 1  # init velocity x
        self._kalman.statePre[3][0] = 1  # init velocity y

        self._kalman.statePost = np.zeros((4, 1), dtype=np.float32)

        self._kalman.statePost[0] = self._point[0]
        self._kalman.statePost[1] = self._point[1]

        self._kalman.measurementMatrix = cv2.setIdentity(self._kalman.measurementMatrix)

        self._kalman.processNoiseCov = np.float32(
            [[pow(self._delta_time, 4.0) / 4.0, 0, pow(self._delta_time, 3.0) / 2.0, 0],
             [0, pow(self._delta_time, 4.0) / 4.0, 0, pow(self._delta_time, 3.0) / 2.0],
             [pow(self._delta_time, 3.0) / 2.0, 0, pow(self._delta_time, 2.0), 0],
             [0, pow(self._delta_time, 3.0) / 2.0, 0, pow(self._delta_time, 2.0)]])

        self._kalman.processNoiseCov *= self._accel_noise_mag
        self._kalman.measurementNoiseCov = cv2.setIdentity(self._kalman.measurementNoiseCov, 0.1)
        self._kalman.errorCovPost = cv2.setIdentity(self._kalman.errorCovPost, .1)

    @staticmethod
    # track_points is an array of PointState object
    def track_points(prev_frame, curr_frame, curr_landmarks, track_points):
        prev_landmarks = []
        curr_landmarks = np.float32(curr_landmarks)

        for el in track_points:
            prev_landmarks.append(el.get_point())

        prev_landmarks = np.float32(prev_landmarks)

        """
        lk_params = dict(criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01), winSize=(7, 7),
                         maxLevel=3,
                         flags=0, minEigThreshold=0.001)
        """

        lk_params = dict(criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), winSize=(15, 15),
                         maxLevel=2)

        p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_landmarks, None, **lk_params)
        for i in range(len(_st)):
            if _st[i]:
                # #print("normale, err: ",_err[i])
                track_points[i].update((p1[i] + curr_landmarks[i]) / 2)

            else:
                track_points[i].update(curr_landmarks[i])

        return track_points


def _crop_face(img, face_loc, padding_size=0):
    '''
    crop face into small image, face only, but the size is not the same
    '''
    h, w, c = img.shape
    top = face_loc[0] - padding_size
    right = face_loc[1] + padding_size
    down = face_loc[2] + padding_size
    left = face_loc[3] - padding_size

    if top < 0:
        top = 0
    if right > w - 1:
        right = w - 1
    if down > h - 1:
        down = h - 1
    if left < 0:
        left = 0
    img_crop = img[top:down, left:right]
    return img_crop


def get_eyes_nose_open_face(csv_path):
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        landmarks_final = []

        positions_left = [41, 42, 43, 44, 45, 46]
        positions_right = [47, 48, 49, 50, 51, 52]
        discarded_frames = []
        for row in csv_reader:
            landmarks = []
            if line_count != 0:
                #print(f'Column names are {", ".join(row)}')
            # else:

                if int(row[4]) == 1:
                    for i in positions_left:
                        landmarks.append((int(float(row[i])), int(float(row[i + 68]))))

                    for i in positions_right:
                        landmarks.append((int(float(row[i])), int(float(row[i + 68]))))

                    landmarks.append((int(float(row[35])), int(float(row[35 + 68]))))
                    landmarks.append((int(float(row[62])), int(float(row[62 + 68]))))

                    landmarks_final.append(landmarks)
                else:
                    landmarks_final.append(landmarks)

                    discarded_frames.append(line_count - 1)

            line_count += 1
        #print(discarded_frames)
    return landmarks_final, discarded_frames


def align_openface(input_dir, out_dir_aligned, target_shape=400):
    command = ['C:/Users/robpa/Desktop/Roberto/OpenFace_2.2.0_win_x64/FeatureExtraction.exe', '-fdir', input_dir,
               '-out_dir', out_dir_aligned, '-simsize', str(target_shape),
               '-format_aligned', 'png', '-nomask', '-wild', '-multiview', '1', '-simalign', '-2Dfp']
    #print("OPENFACEEEE CALL")
    subprocess.Popen(command).wait()
    #print("OPENFACEEEE ENDL")

    dest = out_dir_aligned + "/frames_aligned"
    frames_aligned = []
    #print("ITERATE")

    for frame_name in sorted(os.listdir(dest)):

        # frames_dir = os.path.dirname(dest)
        frame = cv2.imread(dest + '/' + frame_name)
        if frame is not None:
            frames_aligned.append(frame)
        # os.remove(frames_dir + '/' + frames)

    return frames_aligned


def face_landmarks_tracking_frames(frames, input_dir_frames, out_openface, alignment_method=1):
    #fd = FaceDetector()
    steps = [0.5, 0.6, 0.7]

    fd = FaceDetector()

    fa = FaceAligner()
    prev_frame = None

    aligned_frames_openface = align_openface(input_dir_frames, out_openface)
    open_landmarks_all_frames = get_eyes_nose_open_face(out_openface + "/frames.csv")[0]

    # array of PointState objects
    track_points = []

    for index, frame in enumerate(frames):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # detection
        faces = fd.detect(frame)
        f = findRelevantFace(faces, frame.shape[1], frame.shape[0])

        if (f is not None) and (f['img'].size != 0):
            # f['roi'] = enclosing_square(f['roi'])

            landmarks = fa.get_landmarks(frame, f['roi'])
            #landmarks = fd.get_landmarks()
            if prev_frame is None:
                track_points.clear()

                for lp in landmarks:
                    track_points.append(PointState(lp))

            else:
                if len(track_points) == 0:
                    for lp in landmarks:
                        track_points.append(PointState(lp))

                else:
                    track_points = PointState.track_points(prev_frame, frame_gray, landmarks, track_points)

            prev_frame = frame_gray

            # alignment

            #: Landmark indices corresponding to the inner eyes and bottom lip.
            INNER_EYES_AND_BOTTOM_LIP = [3, 6, 13]

            #: Landmark indices corresponding to the outer eyes and nose.
            OUTER_EYES_AND_NOSE = [0, 11, 12]

            tp_landmarks = []
            for tp in track_points:
                tp_landmarks.append(tp.get_point())

            # align1
            if alignment_method == 1:
                aligned_face_opt_kalman = align(400, frame, None, tp_landmarks, INNER_EYES_AND_BOTTOM_LIP)
                aligned_face_without = align(400, frame, None, landmarks, INNER_EYES_AND_BOTTOM_LIP)
                aligned_face_open_landmark = align(400, frame, None, open_landmarks_all_frames[index],
                                                   from_openface=True)

            else:
                leftEyePts = [landmarks[36],landmarks[39]]
                rightEyePts = [landmarks[42],landmarks[45]]
                nose = np.array(landmarks[33])

                landmarks_for_align2 = [leftEyePts, rightEyePts, nose]

                '''
                cv2.circle(frame, (leftEyePts[0][0],leftEyePts[0][1]), 2, (0, 255, 0), -1)
                cv2.circle(frame, (leftEyePts[1][0],leftEyePts[1][1]), 2, (0, 255, 0), -1)
                cv2.circle(frame, (rightEyePts[0][0], rightEyePts[0][1]), 2, (0, 255, 0), -1)
                cv2.circle(frame, (rightEyePts[1][0], rightEyePts[1][1]), 2, (0, 255, 0), -1)


                cv2.imshow("frame", frame)
                cv2.waitKey(0)
                '''



                leftEyePts_tp = [tp_landmarks[36],tp_landmarks[39]]
                rightEyePts_tp = [tp_landmarks[42],tp_landmarks[45]]
                nose_tp = np.array(tp_landmarks[33])
                tp_for_align2 = [leftEyePts_tp, rightEyePts_tp, nose_tp]

                try:
                    leftEyePts_openface = np.array(
                        [open_landmarks_all_frames[index][0], open_landmarks_all_frames[index][3]])
                    rightEyePts_openface = np.array(
                        [open_landmarks_all_frames[index][6], open_landmarks_all_frames[index][9]])
                    nose_openface = np.array(open_landmarks_all_frames[index][12])
                    landmarks_openface_align2 = [leftEyePts_openface, rightEyePts_openface, nose_openface]
                except:
                    #todo sete openface landmarks in case of failure
                    landmarks_openface_align2 = tp_for_align2

                if alignment_method == 2:
                    aligned_face_opt_kalman = align2(frame, tp_for_align2, fd=fd)
                    aligned_face_without = align2(frame, landmarks_for_align2, fd=fd)

                    """
                    cv2.circle(frame, (leftEyePts_openface[0][0],leftEyePts_openface[0][1]), 2, (0, 255, 0), -1)
                    cv2.circle(frame, (leftEyePts_openface[1][0],leftEyePts_openface[1][1]), 2, (0, 255, 0), -1)
                    cv2.circle(frame, (rightEyePts_openface[0][0], rightEyePts_openface[0][1]), 2, (0, 255, 0), -1)
                    cv2.circle(frame, (rightEyePts_openface[1][0], rightEyePts_openface[1][1]), 2, (0, 255, 0), -1)
                    cv2.circle(frame,(nose_openface[0],nose_openface[1]),2, (0, 255, 0), -1)


                    cv2.imshow("frame", frame)
                    cv2.waitKey(0)
                    """
                    aligned_face_open_landmark = align2(frame, landmarks_openface_align2, fd=fd)

                else:

                    aligned_face_opt_kalman = align3(frame, tp_for_align2, f['roi'], fd=fd)
                    aligned_face_without = align3(frame, landmarks_for_align2, f['roi'], fd=fd)

                    aligned_face_open_landmark = align3(frame, landmarks_openface_align2, f['roi'], fd=fd)

            for tp in track_points:
                cv2.circle(frame, tp.get_point(), 2, (0, 255, 0), -1)
            # #print("\n\n")
            for land in landmarks:
                cv2.circle(frame, land, 2, (255, 0, 0), -1)

            for land in open_landmarks_all_frames[index]:
                cv2.circle(frame, land, 2, (0, 0, 255), -1)

            cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 1, (0, 0, 255), -1)

            cv2.imshow("frame before alignment", frame)

            cv2.imshow("OPT + KALMAN", aligned_face_opt_kalman)

            cv2.imshow("NO TEMPORAL", aligned_face_without)

            cv2.imshow("ONLY OPENFACE", aligned_frames_openface[index])

            cv2.imshow("OPENFACE LANDMARK", aligned_face_open_landmark)

            cv2.waitKey(50)
            # #print("ciao")


def extract_frames(src, dest, asr, extract=False):
    if extract is True:
        command = ['ffmpeg', '-loglevel', 'error', '-hide_banner', '-nostats', '-i', src, '-s', asr, '-q:a', '1', dest]
        subprocess.call(command)
    frames_video = []

    for frames in sorted(os.listdir(os.path.dirname(dest))):

        frames_dir = os.path.dirname(dest)
        frame = cv2.imread(frames_dir + '/' + frames)
        if frame is not None:
            frames_video.append(frame)
            # os.remove(frames_dir + '/' + frames)

    return frames_video


TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

INV_TEMPLATE = np.float32([
    (-0.04099179660567834, -0.008425234314031194, 2.575498465013183),
    (0.04062510634554352, -0.009678089746831375, -1.2534351452524177),
    (0.0003666902601348179, 0.01810332406086298, -0.32206331976076663)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


def align(imgDim, rgbImg, bb=None,
          landmarks=None, landmarkIndices=None,
          skipMulti=False, scale=0.9, from_openface=False):
    """align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)

    Transform and align a face in an image.

    :param imgDim: The edge length in pixels of the square the image is resized to.
    :type imgDim: int
    :param rgbImg: RGB image to process. Shape: (height, width, 3)
    :type rgbImg: numpy.ndarray
    :param bb: Bounding box around the face to align. \
               Defaults to the largest face.
    :type bb: dlib.rectangle
    :param landmarks: Detected landmark locations. \
                      Landmarks found on `bb` if not provided.
    :type landmarks: list of (x,y) tuples
    :param landmarkIndices: The indices to transform to.
    :type landmarkIndices: list of ints
    :param skipMulti: Skip image if more than one face detected.
    :type skipMulti: bool
    :param scale: Scale image before cropping to the size given by imgDim.
    :type scale: float
    :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
    :rtype: numpy.ndarray
    """

    INNER_EYES_AND_BOTTOM_LIP_DLIB = [39, 42, 57]
    npLandmarkIndices_dlib = np.array(INNER_EYES_AND_BOTTOM_LIP_DLIB)

    if from_openface:
        INNER_EYES_AND_BOTTOM_LIP = [3, 6, 13]
        npLandmarkIndices2 = np.array(INNER_EYES_AND_BOTTOM_LIP)





    else:
        npLandmarkIndices2 = np.array(npLandmarkIndices_dlib)

    npLandmarks = np.float32(landmarks)

    """
    cv2.circle(rgbImg, (npLandmarks[npLandmarkIndices2[0]][0], npLandmarks[npLandmarkIndices2[0]][1]), 2, (0, 255, 0), -1)
    cv2.circle(rgbImg, (npLandmarks[npLandmarkIndices2[1]][0], npLandmarks[npLandmarkIndices2[1]][1]), 2, (0, 255, 0), -1)
    cv2.circle(rgbImg, (npLandmarks[npLandmarkIndices2[2]][0], npLandmarks[npLandmarkIndices2[2]][1]), 2, (0, 255, 0), -1)
    """

    try:
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices2],
                                   imgDim * MINMAX_TEMPLATE[npLandmarkIndices_dlib] * scale + imgDim * (1 - scale) / 2)
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))

    except Exception as e:
        thumbnail = rgbImg

    return thumbnail


def _angle_between_2_pt(p1, p2):
    '''
    to calculate the angle rad by two points
    '''
    x1, y1 = p1
    x2, y2 = p2
    tan_angle = (y2 - y1) / (x2 - x1)
    return (np.degrees(np.arctan(tan_angle)))


def _get_rotation_matrix(left_eye_pt, right_eye_pt, nose_center, img, scale):
    '''
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    '''
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    M = cv2.getRotationMatrix2D((nose_center[0] / 2, nose_center[1] / 2), eye_angle, scale)

    return M


def align2(img, landmarks, scale=1.5, face_size=(400, 400), fd=None):
    #     '''
    #      landmarks left,right, center nose
    #     face alignment API for single image, get the landmark of eyes and nose and do warpaffine transformation
    #     :param face_img: single image that including face, I recommend to use dlib frontal face detector
    #     :param scale: scale factor to judge the output image size
    #     :return: an aligned single face image
    #     '''
    h, w, c = img.shape
    left_eye_center = np.mean(landmarks[0],axis=0)
    right_eye_center = np.mean(landmarks[1],axis=0)
    nose_center = landmarks[2]
    trotate = _get_rotation_matrix(left_eye_center, right_eye_center, nose_center, img, scale=scale)
    warped = cv2.warpAffine(img, trotate, (w, h))

    # faces = fd.detect(warped)
    # f = findRelevantFace(faces, warped.shape[1], warped.shape[0])
    #
    # if (f is not None) and (f['img'].size != 0):
    #     f['roi'] = enclosing_square(f['roi'])
    #
    #     cutted = cut(warped, f['roi'])
    #     warped = cv2.resize(cutted, face_size)
    # #

    warped = cv2.resize(warped,face_size)
    return warped


##from medium
def get_eyes_nose(eyes, nose):
    left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
    left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
    right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
    right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)
    nose_x = int(nose[0][0] + nose[0][2] / 2)
    nose_y = int(nose[0][1] + nose[0][3] / 2)

    return (nose_x, nose_y), (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal


def rotate_opencv(img, nose_center, angle):
    M = cv2.getRotationMatrix2D(nose_center, angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return rotated


def align3(img, landmarks, roi, fd, mode=True, face_size=400):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if len(roi) > 0:

        x = roi[0]
        y = roi[1]
        w = roi[2] + roi[0]
        h = roi[3] + roi[1]

        left_eye = (landmarks[0][0], landmarks[0][1])
        right_eye = (landmarks[1][0], landmarks[1][1])
        nose = landmarks[2]

        left_eye = landmarks[0]

        right_eye = landmarks[1]

        left_eye_center = np.mean(landmarks[0], axis=0)
        right_eye_center = np.mean(landmarks[1], axis=0)

        center_of_forehead = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
        center_pred = (int((x + w) / 2), int((y + y) / 2))
        length_line1 = distance(center_of_forehead, nose)
        length_line2 = distance(center_pred, nose)
        length_line3 = distance(center_pred, center_of_forehead)
        cos_a = cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)
        rotated_point = rotate_point(nose, center_of_forehead, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
        if is_between(nose, center_of_forehead, center_pred, rotated_point):
            angle = np.degrees(-angle)
        else:
            angle = np.degrees(angle)

        if mode:
            img = rotate_opencv(img, (nose[0],nose[1]), angle)
        else:
            img = Image.fromarray(img)
            img = np.array(img.rotate(angle))

        faces = fd.detect(img)
        f = findRelevantFace(faces, img.shape[1], img.shape[0])

        if (f is not None) and (f['img'].size != 0):
            f['roi'] = enclosing_square(f['roi'])

            cutted = cut(img, f['roi'])
            img = cv2.resize(cutted, (face_size, face_size))
        #

    return img


if __name__ == "__main__":
    input_dir_frames = "C:/Users/robpa/Desktop/Roberto/Thesis/frames"
    path_to_avis = "C:/Users/robpa/Desktop/Roberto/Thesis/project/videos"
    out_dir_openface = "C:/Users/robpa/Desktop/Roberto/Thesis/aligned_open"

    for video in os.listdir(path_to_avis):
        if os.path.exists(input_dir_frames):
            shutil.rmtree(input_dir_frames)
            os.makedirs(input_dir_frames)

        else:
            os.makedirs(input_dir_frames)

        if os.path.exists(out_dir_openface):
            shutil.rmtree(out_dir_openface)
            os.makedirs(out_dir_openface)

        else:
            os.makedirs(out_dir_openface)
        frames = extract_frames(path_to_avis + "/" + video, input_dir_frames + "//frame-%3d.png", "720x480", True)
        face_landmarks_tracking_frames(frames, input_dir_frames, out_dir_openface, 2)
        # out_dir = input_dir+"/processed"
