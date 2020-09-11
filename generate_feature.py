import subprocess
import sys
import glob
import librosa
import numpy as np

from tqdm import tqdm
from os import listdir, path
from keras.models import Model

from keras_yamnet.preprocessing import preprocess_input
from keras_yamnet.yamnet import YAMNet

root_path = "/user/vlongobardi/"


def to_milliseconds(t_stamp):
    t = t_stamp.split(":")
    m = int(t[1])
    s_ms = t[2].split(".")
    return int(s_ms[1]) * 10 + int(s_ms[0]) * 1000 + m * 6000


def to_t_stamp(milliseconds):
    s = milliseconds // 1000
    milliseconds = milliseconds % 1000
    string_sec = str(s)
    if milliseconds < 10:
        zero = "00"
    elif milliseconds < 100:
        zero = "0"
    else:
        zero = ""
    if len(string_sec) == 1:
        string_sec = "0" + string_sec
    return "00:00:" + string_sec + "." + zero + str(milliseconds)


def generate_files(base_dir):
    """
    Generate an iterator that returns:
    - Dataset name: "Train" or "Val"
    - Emotion: "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
    - Name: name of the video clip
    """
    for dataset in listdir(base_dir):
        for emotion in [emo for emo in listdir(base_dir + "Train/") if path.isdir(base_dir + "Train/" + emo)]:
            if path.isdir(base_dir + "Train/" + emotion):
                for video in listdir(base_dir + dataset + "/" + emotion):
                    yield dataset + "/" + emotion + "/" + video.split(".")[0]


# ffmpeg -i /Val/Disgust/000738334.avi -ab 128k -ac 2 -ar 48000 -vn temp_wav/Val/Disgust/000738334.avi
def from_avi_to_wav():
    """ For each video-clip generate a corresponding audio-clip """
    avi_dir = root_path + "AFEW/videos/"
    wav_dir = root_path + "temp_wav/"
    for file_path in tqdm(generate_files(avi_dir)):
        cmd = "ffmpeg -i " + avi_dir + file_path + ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir + file_path + ".wav"
        # #print(name, "\n\n\n")
        subprocess.call(cmd, shell=True)


# ffmpeg -y -i ~/AFEW/videos/Train/Angry/000046280.avi 2>&1 | grep Duration | awk '{#print $2}' | tr -d ,
# ffmpeg -ss 0.3 -i ~/AFEW/videos/Train/Angry/000046280.avi -t 0.3 -ab 128k -ac 2 -ar 48000 -vn 000046280_0.wav
def execute_command(wav_dir, file_path, frame_size, clip_dir, i, offset=0):
    if offset:
        ss_option = " -ss " + to_t_stamp(offset)
    else:
        ss_option = ""

    cmd = "ffmpeg -y" + ss_option + " -i " + wav_dir + file_path + ".wav -t " + to_t_stamp(
        frame_size) + " -ab 128k -ac 2 -ar 48000 -vn " + clip_dir + file_path + "_" + str(i) + ".wav"
    # print(cmd, "\n")
    subprocess.call(cmd, shell=True)


def from_wav_to_clips(frame_size=300, frame_step=150, offset=0):
    """ For each audio-clip generate a sub-audio-clips with specific frame size and overlapping """
    wav_dir = root_path + "temp_wav/"
    clip_dir = root_path + "temp_clips_" + str(frame_size) + "/"
    for file_path in tqdm(generate_files(wav_dir)):
        cmd = "ffmpeg -y -i " + wav_dir + file_path + ".wav temp_output.wav"

        bash_output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # print("\n\n", bash_output.stderr.decode("utf-8"), "\n\n")
        duration = to_milliseconds(bash_output.stderr.decode("utf-8").split("Duration: ")[1].split(",")[0])
        i = 0
        execute_command(wav_dir, file_path, frame_size, clip_dir, i, offset)
        d0 = frame_step + offset
        while d0 < (duration - frame_size):
            i += 1
            execute_command(wav_dir, file_path, frame_size, clip_dir, i, d0)
            d0 += frame_step
        if d0 < duration and offset == 0:
            execute_command(wav_dir, file_path, frame_size, clip_dir, i + 1, duration - frame_size)


# SMILExtract -C opensmile-2.3.0/config/emobase2010_2.conf -I test.wav -O output.arff -instname input
def from_clips_to_feature(cfg_file="emobase2010.conf", frame_size=300, middle_feature_dir="", base_dir=None):
    if base_dir is None:
        base_dir = root_path + "temp_clips/"
    feature_dir = root_path + middle_feature_dir + cfg_file.split(".")[0] + "_" + str(frame_size) + "/"
    config_path = root_path + "opensmile-2.3.0/config/" + cfg_file
    for file_path in tqdm(generate_files(base_dir)):
        cmd = "SMILExtract -C " + config_path + " -I " + base_dir + file_path + ".wav -O " + feature_dir + file_path + \
              ".arff"
        subprocess.call(cmd, shell=True)


def get_winsize_featurenum(weights):
    if "full" in weights:
        return 620, "full"
    elif "1000" in weights:
        return 98, "1000"
    elif "600" in weights:
        return 58, "600"
    elif "300" in weights:
        return 28, "300"
    elif "100" in weights:
        return 8, "100"
    else:
        raise Exception("Window dimension unexpected.")


def generate_feature_yamnet(model, files, win_dim, feature_number, mode="Train"):
    execeptions = []
    flag_except = 0
    for wav in tqdm(files):
        try:
            c = wav.split("/")[-2]
            clip_id = wav.split("/")[-1].split(".")[0]
            signal, sound_sr = librosa.load(wav, 48000)
            if win_dim == "full" and len(signal) < 298368:  # max_length
                mul = np.tile(signal, 298368 // len(signal))
                add = signal[:298368 % len(signal)]
                signal = np.concatenate([mul, add])
            mel = preprocess_input(signal, sound_sr)
            mel = mel.reshape(1, feature_number, 64)
            feature = model.predict(mel)
            pth = root_path + "early_feature/emobase2010_" + win_dim + "_yam/" + mode + "/" + c + "/" + clip_id + ".npy"
            np.save(pth, feature)
        except ValueError:
            execeptions.append(wav)
            print("wav", wav)
    if flag_except:
        print("\n\n\nexeceptions", len(execeptions), execeptions)


def yamnet_extract(weights):
    feature_number, win_size = get_winsize_featurenum(weights)
    m = YAMNet(classes=7, classifier_activation='softmax', input_shape=(feature_number, 64))
    m.load_weights(weights)
    model = Model(input=m.input, output=m.layers[-3].output)

    train_wavs = glob.glob(root_path + "early_feature/temp_clips_" + win_size + "/Train/*/*.wav")
    val_wavs = glob.glob(root_path + "early_feature/temp_clips_" + win_size + "/Val/*/*.wav")

    generate_feature_yamnet(model, train_wavs, win_size, feature_number, "Train")
    generate_feature_yamnet(model, val_wavs, win_size, feature_number, "Val")


if __name__ == "__main__":
    fs = int(sys.argv[1])
    from_wav_to_clips(fs)
    if sys.argv[2] == "e":
        cfg = "emobase2010.conf"
    else:
        cfg = "IS09_emotion.conf"
    from_clips_to_feature(cfg, fs)
