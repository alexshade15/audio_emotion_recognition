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
    for dataset in listdir(base_dir):
        for emotion in [emo for emo in listdir(base_dir + "Train/") if path.isdir(base_dir + "Train/" + emo)]:
            if path.isdir(base_dir + "Train/" + emotion):
                for video in listdir(base_dir + dataset + "/" + emotion):
                    yield dataset, emotion, video.split(".")[0]


def get_avi_info():
    base_dir = "/user/vlongobardi/AFEW/videos/"

    for dataset, emotion, name in generate_files(base_dir):
        command = "ffmpeg -i " + base_dir + dataset + "/" + emotion + "/" + name + ".avi"
        subprocess.call(command, shell=True)
        print(base_dir + dataset + "/" + emotion + "/" + name, "\n\n\n")


# ffmpeg -y -i /user/vlongobardi/AFEW/videos/Train/Angry/000046280.avi 2>&1 | grep Duration | awk '{print $2}' | tr -d ,
# ffmpeg -ss 0 -t 0.3 -i /user/vlongobardi/AFEW/videos/Train/Angry/000046280.avi -ab 128k -ac 2 -ar 48000 -vn 000046280_0.wav
def from_avi_to_wav():
    base_dir = "/user/vlongobardi/AFEW/videos/"
    wav_dir = "/user/vlongobardi/wav/"
    frame_size = 300
    frame_step = 150
    for dataset, emotion, name in generate_files(base_dir):
        command = "ffmpeg -y -i " + base_dir + dataset + "/" + emotion + "/" + name + ".avi temp_output.wav"
        bash_output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        duration = to_milliseconds(bash_output.stderr.decode("utf-8").split("Duration: ")[1].split(",")[0])
        i = 0
        d0 = frame_step

        command = "ffmpeg -t " + to_t_stamp(frame_size) + " -i " + base_dir + dataset + "/" + emotion + "/" + name + \
                  ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir + dataset + "/" + emotion + "/" + name + "_" + str(i) \
                  + ".wav"
        subprocess.call(command, shell=True)

        while d0 < (duration - frame_size):
            i += 1
            command = "ffmpeg -ss " + to_t_stamp(d0) + " -t " + to_t_stamp(frame_size) + " -i " + base_dir + dataset + \
                      "/" + emotion + "/" + name + ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir + dataset + "/" + \
                      emotion + "/" + name + "_" + str(i) + ".wav"
            subprocess.call(command, shell=True)
            d0 += frame_step
        if d0 < duration:
            command = "ffmpeg -ss " + to_t_stamp(duration - frame_size) + " -t " + to_t_stamp(frame_size) + " -i " + \
                      base_dir + dataset + "/" + emotion + "/" + name + ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir \
                      + dataset + "/" + emotion + "/" + name + "_" + str(i+1) + ".wav"
            subprocess.call(command, shell=True)


def from_wav_to_feature():
    base_dir = "/user/vlongobardi/wav/"
    feature_dir = "/user/vlongobardi/audio_feature/"
    config = "/user/vlongobardi/opensmile-2.3.0/config/emobase2010_2.conf"
    # SMILExtract -C opensmile-2.3.0/config/emobase2010_2.conf -I test.wav -O output.arff -instname input
    for dataset, emotion, name in generate_files(base_dir):
        command = "SMILExtract -C " + config + " -I " + base_dir + dataset + "/" + emotion + "/" + name + ".wav -O " + feature_dir + dataset + "/" + emotion + "/" + name + ".arff -instname " + name
        print(name, "\n\n\n")
        subprocess.call(command, shell=True)

# from_avi_to_wav()
# from_wav_to_feature()
