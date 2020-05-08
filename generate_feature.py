import subprocess
from os import listdir, path


def to_milliseconds(t_stamp):
    t = t_stamp.split(":")
    # h = int(t[0])
    m = int(t[1])
    s_ms = t[2].split(".")
    return int(s_ms[1]) * 10 + int(s_ms[0]) * 1000 + m * 6000  # + h * 360000


def to_t_stamp(milliseconds):
    s = milliseconds // 6000
    milliseconds = milliseconds % 6000
    string_sec = str(s)
    if len(string_sec) == 1:
        string_sec = "0" + string_sec
    return "00:00:" + string_sec + "." + str(milliseconds)


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


def get_avi_info():
    base_dir = "~/AFEW/videos/"
    for dataset, emotion, name in generate_files(base_dir):
        command = "ffmpeg -i " + base_dir + dataset + "/" + emotion + "/" + name + ".avi"
        subprocess.call(command, shell=True)
        print(base_dir + dataset + "/" + emotion + "/" + name, "\n\n\n")


def from_avi_to_wav():
    avi_dir = "~/AFEW/videos/"
    wav_dir = "~/temp_wav/"
    for file_path in generate_files(avi_dir):
        cmd = "ffmpeg -i " + avi_dir + file_path + ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir + file_path + ".wav"
        # print(name, "\n\n\n")
        subprocess.call(cmd, shell=True)


# ffmpeg -y -i ~/AFEW/videos/Train/Angry/000046280.avi 2>&1 | grep Duration | awk '{print $2}' | tr -d ,
# ffmpeg -ss 0.3 -i ~/AFEW/videos/Train/Angry/000046280.avi -t 0.3 -ab 128k -ac 2 -ar 48000 -vn 000046280_0.wav
def from_wav_to_clips():
    wav_dir = "~/temp_wav/"
    clip_dir = "~/temp_clips/"
    frame_size = 300
    frame_step = 150
    for file_path in generate_files(wav_dir):
        cmd = "ffmpeg -y -i " + wav_dir + file_path + ".avi temp_output.wav"
        bash_output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        duration = to_milliseconds(bash_output.stderr.decode("utf-8").split("Duration: ")[1].split(",")[0])
        i = 0
        d0 = frame_step
        cmd = "ffmpeg -i " + wav_dir + file_path + ".avi -t " + to_t_stamp(frame_size) + \
              "-ab 128k -ac 2 -ar 48000 -vn " + clip_dir + file_path + "_" + str(i) + ".wav"
        # print("\n\n", cmd)
        subprocess.call(cmd, shell=True)
        # if duration == 480:
        #     continue

        while d0 < (duration - frame_size):
            # if duration in {550, 600, 620}:
            #     d0 += frame_step
            #     break
            i += 1
            cmd = "ffmpeg -ss " + to_t_stamp(d0) + " -i " + wav_dir + file_path + ".avi -t " + to_t_stamp(frame_size) \
                  + " -ab 128k -ac 2 -ar 48000 -vn " + clip_dir + file_path + "_" + str(i) + ".wav"
            # print("\n\n", cmd)
            subprocess.call(cmd, shell=True)
            d0 += frame_step
        if d0 < duration:
            cmd = "ffmpeg -ss " + to_t_stamp(duration - frame_size) + " -i " + wav_dir + file_path + ".avi -t " + \
                  to_t_stamp(frame_size) + "-ab 128k -ac 2 -ar 48000 -vn " + clip_dir + file_path + "_" + str(i + 1) \
                  + ".wav"
            # print("\n\n", cmd)
            subprocess.call(cmd, shell=True)


def from_clips_to_feature(cfg_file):
    base_dir = "~/temp_clips/"
    feature_dir = "~/audio_feature_" + cfg_file + "/"
    config_path = "~/opensmile-2.3.0/config/" + cfg_file
    # SMILExtract -C opensmile-2.3.0/config/emobase2010_2.conf -I test.wav -O output.arff -instname input
    for file_path in generate_files(base_dir):
        cmd = "SMILExtract -C " + config_path + " -I " + base_dir + file_path + ".wav -O " + feature_dir + file_path + \
              ".arff"  # -instname " + name
        subprocess.call(cmd, shell=True)
