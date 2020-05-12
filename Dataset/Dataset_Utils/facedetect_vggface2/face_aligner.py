import cv2
import numpy as np
import os
import dlib
from PIL import Image

PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "models", "shape_predictor_68_face_landmarks.dat")


def _get_part(shape, n):
    l = shape.part(n)
    return (l.x, l.y)


class FaceAligner:
    predictor = None

    def __init__(self):
        print("FaceAligner -> init")
        self.predictor = dlib.shape_predictor(PATH)
        print("FaceAligner -> init ok")

    def align(self,image,roi):
        landmarks = self.get_landmarks(image, roi)
        desiredFaceWidth = 256
        desiredFaceHeight = 256
        desiredLeftEye = [0.365, 0.365]

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        leftEyePts = np.array([landmarks[0],landmarks[1]])
        rightEyePts = np.array([landmarks[2],landmarks[3]])
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        leftEyeCenter = tuple(leftEyeCenter)
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        rightEyeCenter = tuple(rightEyeCenter)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))


        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2, (leftEyeCenter[1] + rightEyeCenter[1]) / 2)

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)

        return output

    def get_landmarks(self, image, box):
        box = dlib.rectangle(box[0], box[1], box[0]+box[2], box[1]+box[3])
        shape = self.predictor(image, box)
        arr = []
        for i in range(64):
            arr.append(_get_part(shape, i))
        return [_get_part(shape,36),_get_part(shape,39),_get_part(shape,42), _get_part(shape,45), _get_part(shape,33), _get_part(shape,66)]
        # return [_get_part(shape,36), _get_part(shape,45), _get_part(shape,33), _get_part(shape,66)]

    def get_shape_detections(self,image,box):
        return self.predictor(image, box)
        
    def __del__(self):
        print("FaceAligner -> bye")
