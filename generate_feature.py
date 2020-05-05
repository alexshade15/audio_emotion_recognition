import subprocess
from os import listdir, path


def to_milliseconds(t_stamp):
    t = t_stamp.split(":")
    h = int(t[0])
    m = int(t[1])
    s_ms = t[2].split(".")
    return int(s_ms[1]) * 10 + int(s_ms[0]) * 1000 + m * 6000 + h * 360000


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


def from_avi_to_wav():
    base_dir = "/user/vlongobardi/AFEW/videos/"
    wav_dir = "/user/vlongobardi/wav/"
    frame_size = 300
    frame_step = 150
    for dataset, emotion, name in generate_files(base_dir):
        command = "ffmpeg -y -i " + base_dir + dataset + "/" + emotion + "/" + name + ".avi temp_output.wav"
        #print(command, "\n\n")
        bash_output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        #print("\n\n--", type(bash_output), bash_output.stdout, bash_output.stderr)
        duration = to_milliseconds(bash_output.stderr.decode("utf-8").split("Duration: ")[1].split(",")[0])
        i = 0
        d0 = 0
        while d0 < (duration - frame_size):
            command = "ffmpeg -ss " + str(d0) + " -t " + str(d0 + frame_size) + " -i " + base_dir + dataset + "/" + emotion + "/" + name + ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir + dataset + "/" + emotion + "/" + name + "_" + str(i) + ".wav"
            subprocess.call(command, shell=True)
            print(base_dir + dataset + "/" + emotion + "/" + name + "_" + str(i), "\n\n\n")
            i += 1
            d0 += frame_step
        if d0 < duration:
            command = "ffmpeg -ss " + str(duration - frame_size) + " -t " + str(duration) + " -i " + base_dir + dataset + "/" + emotion + "/" + name + ".avi -ab 128k -ac 2 -ar 48000 -vn " + wav_dir + dataset + "/" + emotion + "/" + name + "_" + str(i) + ".wav"
            subprocess.call(command, shell=True)
            print(base_dir + dataset + "/" + emotion + "/" + name + "_" + str(i), "\n\n\n")


def from_wav_to_feature():
    base_dir = "/user/vlongobardi/wav/"
    feature_dir = "/user/vlongobardi/audio_feature/"
    config = "/user/vlongobardi/opensmile-2.3.0/config/emobase2010_2.conf"
    # SMILExtract -C opensmile-2.3.0/config/emobase2010_2.conf -I test.wav -O output.arff -instname input
    for dataset, emotion, name in generate_files(base_dir):
        command = "SMILExtract -C " + config + " -I " + base_dir + dataset + "/" + emotion + "/" + name + ".wav -O " + feature_dir + dataset + "/" + emotion + "/" + name + ".arff -instname " + name
        print(name, "\n\n\n")
        subprocess.call(command, shell=True)


#from_avi_to_wav()
#from_wav_to_feature()
