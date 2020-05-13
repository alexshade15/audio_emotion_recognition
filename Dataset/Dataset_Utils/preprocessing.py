from Dataset.Dataset_Utils.dataset_tools import equalize_hist, linear_balance_illumination, mean_std_normalize
import numpy as np
import cv2

def preprocessing_no(img):
    return img
def preprocessing_full(img, impath):
    img = equalize_hist(img)
    img = img.astype(np.float32)
    img = linear_balance_illumination(img)
    
    if np.abs(np.min(img)-np.max(img)) < 1:
        print("WARNING: Image is =%d" % np.min(img), "where:", impath)
        
    else:
        img = mean_std_normalize(img)
    return img
def preprocessing_imagenet(img):
    ds_means = np.array([0.485, 0.456, 0.406])*255
    ds_stds = np.array([0.229, 0.224, 0.225])*255
    return mean_std_normalize(img, ds_means, ds_stds)

def preprocessing_vggface2(img):
    VGGFACE2_MEANS = np.array([131.0912, 103.8827, 91.4953])
    return mean_std_normalize(img, VGGFACE2_MEANS, None)