import random
from Dataset.Dataset_Utils.dataset_tools import _random_normal_crop, random_image_rotate, roi_center, random_image_skew, \
    random_change_image, random_change_roi, enclosing_square


class DefaultAugmenter():
    def __init__(self, target_shape = (224,224,3)):
        self.target_shape = target_shape
        self._init_random_values()

    def _init_random_values(self):
        # param img rotate
        self.random_angle_deg = _random_normal_crop(1, 10)[0]
        # param img skew
        self.random_skew = _random_normal_crop(2, 0.1, positive=True)
        # random img change
        self.a = _random_normal_crop(1, 0.5, mean=1)[0]
        self.b = _random_normal_crop(1, 48)[0]
        self.random_flip = random.randint(0, 1)
        # param roi change
        self.max_change_fraction = 0.008
        sigma = self.target_shape[0] * self.max_change_fraction
        self.xy = _random_normal_crop(2, sigma, mean=-sigma / 5).astype(int)
        self.wh = _random_normal_crop(2, sigma * 2, mean=sigma / 2, positive=False).astype(int)

    def reset_state(self):
        self._init_random_values()

    def before_cut(self, frame, roi):
        frame = random_image_rotate(frame, roi_center(roi), self.random_angle_deg)
        frame = random_image_skew(frame, self.random_skew)
        return frame

    def augment_roi(self, roi):
        roi = random_change_roi(roi, random_values=(self.xy, self.wh))
        roi = enclosing_square(roi)
        return roi

    def after_cut(self, frame):
        frame = random_change_image(frame, random_values=(self.a, self.b, self.random_flip))
        return frame


class NoAug():
    def before_cut(self, frame, roi):
        return frame
    def augment_roi(self, roi):
        return roi
    def after_cut(self, frame):
        return frame
    def reset_state(self):
        pass
