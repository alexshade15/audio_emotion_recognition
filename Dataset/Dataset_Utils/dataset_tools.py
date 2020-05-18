import os
import random
import subprocess
import cv2
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

# AFEW utils


openface_command_feature_extraction = '/data/s4180941/OpenFace/build/bin/FeatureExtraction'
openface_command_faceland_img = '/data/s4180941/OpenFace/build/bin/FaceLandmarkImg'


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


def openface_call(openface_fdir, out_dir, cache_p, partition, bbox=None, as_img=True):
    """preprocess video"""

    # create command for open face
    if as_img:
        command = [openface_command_faceland_img]

    else:
        command = [openface_command_feature_extraction]

    for _dir in openface_fdir:
        command.append("-fdir")
        command.append(_dir)

    if bbox is not None:
        command.append("-bboxdir")
        command.append(bbox)

    # 400 1.46  200 0.73
    resize_shape = 400
    scale = 1.46

    command += ['-out_dir', out_dir, '-simsize', str(resize_shape), '-simscale', str(scale),
                '-format_aligned', 'jpg', '-nomask', '-simalign', '-wild', '-multi_view', '1']

    if not as_img:
        command += ['-nobadaligned']

    try:
        #print("Calling OpenFace")
        log_file = open(os.path.join(cache_p, 'OpenFace_output_' + partition + '.log'), "a")
        p = subprocess.Popen(command, stdout=log_file, stderr=log_file).wait()
        log_file.close()

        # p = subprocess.Popen(command, shell=True).wait()

        #print("End OpenFace")

    except Exception as e:
        print(e)


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
    return width, height


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
    return f['roi'][0], f['roi'][1]


def bottom_right(f):
    return f['roi'][0] + f['roi'][2], f['roi'][1] + f['roi'][3]


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
    original = img.copy()

    img = (img - 128.0) * a + 128.0 + b
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # flip
    if random:
        img = np.fliplr(img)
    if np.array_equal(img, np.zeros((img.shape))) is True or np.max(img) <= 10:
        img = original.copy()
    return img


def random_change_roi(roi, max_change_fraction=0.045, only_narrow=False, random_values=None):
    # random crop con prob + alta su 0 (gaussiana)
    sigma = roi[3] * max_change_fraction
    if random_values is None:
        xy = _random_normal_crop(2, sigma, mean=-sigma / 5).astype(int)
        wh = _random_normal_crop(2, sigma * 2, mean=sigma / 2, positive=only_narrow).astype(int)
    else:
        xy, wh = random_values
    # #print("orig roi: %s" % str(roi))
    # #print("rand changes -> xy:%s, wh:%s" % (str(xy), str(wh)))
    roi2 = (roi[0] + xy[0], roi[1] + xy[1], roi[2] - wh[0], roi[3] - wh[1])
    # #print("new roi: %s" % str(roi2))

    return roi2


def roi_center(roi):
    return (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2)


def random_image_rotate(img, rotation_center, random_angle_deg=_random_normal_crop(1, 10)[0]):
    angle_deg = random_angle_deg
    M = cv2.getRotationMatrix2D(rotation_center, angle_deg, 1.0)
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2])
    return nimg.reshape(img.shape)


def random_image_skew(img, random_skew=_random_normal_crop(2, 0.1, positive=True)):
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
