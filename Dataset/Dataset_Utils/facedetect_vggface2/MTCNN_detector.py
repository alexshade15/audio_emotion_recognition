from mtcnn import MTCNN



class MTCNN_detector():
    net = None

    def __init__(self, steps_threshold = [0.6, 0.7, 0.7]):
        print("FaceDetector MTCNN -> init")
        self.net = MTCNN(steps_threshold=steps_threshold)
        self.min_confidence = steps_threshold
        print("FaceDetector MTCNN -> init ok")

    def detect(self, image):

        detections = self.net.detect_faces(image)
        faces_result = []
        for res in (detections):
            confidence = res['confidence']
            keypoints = list(res['keypoints'].values())
            x1 = res['box'][0]
            y1 = res['box'][1]
            x2 = res['box'][2] + res['box'][0]
            y2 = res['box'][3] + res['box'][1]
            f = (x1, y1, x2 - x1, y2 - y1)
            if f[2] > 1 and f[3] > 1 and f[0] > 0 and f[1] > 0:
                face = ({
                    'roi': f,
                    'type': 'face',
                    'img': image[f[1]:f[1] + f[3], f[0]:f[0] + f[2]],
                    'confidence': confidence,
                    'landmarks': keypoints,
                })
                if face['img'].shape[0] != 0 and face['img'].shape[1] != 0:
                    faces_result.append(face)
                else:
                    pass
        return faces_result

    def get_landmarks(self,detections):

        return detections[0]['keypoints'].values()



    def __del__(self):
        print("FaceDetector -> bye")