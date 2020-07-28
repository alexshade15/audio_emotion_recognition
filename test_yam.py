import numpy as np
import glob
import librosa
import keras
from keras.utils import to_categorical, Sequence
from keras.optimizers import Adagrad
from keras_yamnet.yamnet import YAMNet
from keras_yamnet.preprocessing import preprocess_input
from tqdm import tqdm


class RandomAudioGenerator(Sequence):

    def __init__(self, X_list, y, sample_per_class, batches_per_epoch, win_sec=0.5, sr=16000, transform=None,
                 augment=None):
        self.X_list = X_list
        self.y = y
        self.sample_per_class = sample_per_class
        self.batches_per_epoch = batches_per_epoch
        self.sr = sr
        self.win_sec = win_sec
        self.transform = transform
        self.augment = augment
        self.win_samples = int(sr * win_sec) if transform is None else int(
            (win_sec - transform.win_sec) / transform.hop_sec) + 1
        self.n_classes = len(set(y))
        self.X = []

        # take memory of the files belonging to each class
        self.y_ind = {}

        for i in range(self.n_classes):
            self.y_ind[i] = []

        for i in tqdm(range(len(self.X_list)), desc='Dataset Loading'):
            file_path = self.X_list[i]
            # File loading
            data, sound_sr = librosa.load(file_path, self.sr)
            # Features extraction
            data = data if self.transform is None else self.transform(data, sound_sr)
            self.X.append(data)
            self.y_ind[self.y[i]].append(i)

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):

        X_batch = []
        y_batch = []

        for i in range(self.n_classes):
            for k in range(self.sample_per_class):
                # select random file belonging to the selected class
                file_idx = self.y_ind[i][np.random.randint(len(self.y_ind[i]))]

                # select random frame from the selected file
                start = np.random.randint(self.X[file_idx].shape[0] - self.win_samples)
                end = start + self.win_samples

                data = self.X[file_idx][start:end]
                data = data if self.augment is None else self.augment(data)

                label = to_categorical(i, num_classes=self.n_classes)

                X_batch.append(data)
                y_batch.append(label)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        return X_batch, y_batch


class SequentialAudioGenerator(Sequence):

    def __init__(self, X_list, y, batch_size, win_sec=0.5, hop_sec=0.25, sr=16000, transform=None):
        self.X_list = X_list
        self.y = y
        self.batch_size = batch_size
        self.sr = sr
        self.win_sec = win_sec
        self.win_samples = int(sr * win_sec) if transform is None else int(
            (win_sec - transform.win_sec) / transform.hop_sec) + 1
        self.hop_sec = hop_sec
        self.hop_samples = int(sr * hop_sec) if transform is None else int(
            (hop_sec - transform.win_sec) / transform.hop_sec) + 1
        self.transform = transform

        self.n_classes = len(set(y))

        self.X = []
        self.y_ind = []

        for i in tqdm(range(len(self.X_list)), desc='Dataset Loading'):
            file_path = self.X_list[i]

            data, sound_sr = librosa.load(file_path, self.sr)
            data = data if self.transform is None else self.transform(data, sound_sr)

            for start in range(0, data.shape[0] - self.win_samples, self.hop_samples):
                self.X.append(data[start:start + self.win_samples])

                self.y_ind.append(to_categorical(self.y[i], num_classes=self.n_classes))

        self.batches_per_epoch = len(self.X) // self.batch_size
        self.batches_per_epoch += 1 if len(self.X) % self.batch_size != 0 else 0

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):

        X_batch = self.X[index * self.batch_size:min((index + 1) * self.batch_size, len(self.X))]
        y_batch = self.y_ind[index * self.batch_size:min((index + 1) * self.batch_size, len(self.X))]

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        return X_batch, y_batch


class Extractor:

    def __init__(self):
        self.sr = 48000
        self.win_sec = 0.025
        self.hop_sec = 0.010
        self.win_samples = int(self.sr * self.win_sec)
        self.hop_samples = int(self.sr * self.hop_sec)

    def __call__(self, signal, sound_sr):
        return preprocess_input(signal, sound_sr)


# Data Augmentation
def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.1):
    spec = spec.copy()

    ow = np.min(spec)
    for i in range(num_mask):
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(round(freq_percentage * spec.shape[1]))
        f0 = np.random.uniform(low=0.0, high=spec.shape[1] - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = ow

        time_percentage = np.random.uniform(0.0, freq_masking_max_percentage)

        num_frames_to_mask = int(round(time_percentage * spec.shape[0]))
        t0 = np.random.uniform(low=0.0, high=spec.shape[0] - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = ow

    return spec


def augment(x):
    return spec_augment(x, 2)


def get_data_for_generator(feature_name="emobase2010_600", dataset="Train"):
    base_path = "/user/vlongobardi/late_feature/" + feature_name + "_wav/" + dataset
    datas = glob.glob(base_path + "/*/*.wav")
    return datas, [path.split("/")[-2] for path in datas]


feature_name = "emobase2010_600"
X_train, y_train = get_data_for_generator(feature_name, "Train")
X_val, y_val = get_data_for_generator(feature_name, "Val")

samples_per_class = 5
batches_per_epoch = 100

sr = 48000
win_sec = 0.6  # 0.1, 0.3, 0.6, full
hop_sec = 0.3  # 0.05, 0.15, 0.3, n.a.

transform = Extractor()

train_gen = RandomAudioGenerator(X_train, y_train, samples_per_class, batches_per_epoch, sr=sr, win_sec=win_sec,
                                 transform=transform, augment=augment)
val_gen = SequentialAudioGenerator(X_val, y_val, batch_size=128, sr=sr, win_sec=win_sec, hop_sec=hop_sec,
                                   transform=transform)

model = YAMNet(weights='keras_yamnet/yamnet_conv.h5', classes=7, classifier_activation='softmax')
model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.003, decay=1e-6), metrics=['accuracy'])
model.summary()
callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True)]

model.fit_generator(train_gen, steps_per_epoch=batches_per_epoch, epochs=5, callbacks=callbacks,
                    validation_data=val_gen, shuffle=False, validation_steps=batches_per_epoch)
