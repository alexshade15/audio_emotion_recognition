import glob
import os
import random
import cv2
import keras
import numpy as np
from Dataset.Dataset_Utils.dataset_tools import cut, equalize_hist
from Dataset.Dataset_Utils.preprocessing import preprocessing_full, preprocessing_imagenet, preprocessing_no, preprocessing_vggface2


MAX_TRIES = 3000
IMAGE_EXT = 'jpg'
# input images are 400x400 with the face in the center of size 224x224
DEFAULT_ROI = (88, 88, 224, 224)

CLASSES = ["Angry", "Disgust","Fear","Happy","Neutral","Sad","Surprise"]


def loadAffWild(datadir, anndir):
    videos = {}
    # TODO generalizzare per AFEW
    csvs = glob.glob(os.path.join(datadir, '*.csv'))
    for f in csvs:
        frames_info = []
        vname = os.path.basename(f)[:-4]
        align_info = open(f)
        if '_right' in vname or '_left' in vname:
            try:
                annotation = open(os.path.join(anndir, vname + '.txt'))
            except FileNotFoundError:
                #this check is due to the fact that some double faces video don't contain the face position (left or right)
                #in the annotation name
                vname_position = vname
                vname_position = vname_position.replace('_right','')
                vname_position = vname_position.replace('_left','')
                annotation = open(os.path.join(anndir, vname_position + '.txt'))

        else:
            annotation = open(os.path.join(anndir,vname + '.txt'))
        for i_ali, i_ann in zip(align_info, annotation):
            i_ali = [x.strip() for x in i_ali.split(',')]
            i_ann = [x.strip() for x in i_ann.split(',')]
            if not i_ali[0].isdigit():
                continue
            thisframe = {
                'success': bool(int(i_ali[4])),
                'confidence': float(i_ali[3]),
                'annotation': (float(i_ann[0]), float(i_ann[1]))
            }
            frames_info.append(thisframe)
        
        videos[vname] = frames_info
    return videos


def loadAFEW(datadir):
    videos = {}
    # TODO generalizzare per AFEW

    for label_folder in glob.glob(os.path.join(datadir,'*')):

        csvs = glob.glob(os.path.join(label_folder, '*.csv'))
        for f in csvs:
            frames_info = []
            vname = os.path.basename(f)[:-4]
            align_info = open(f)

            annotation = os.path.basename(label_folder)
            for i_ali in align_info:
                i_ali = [x.strip() for x in i_ali.split(',')]
                if not i_ali[0].isdigit():
                    continue
                thisframe = {
                    'success': bool(int(i_ali[4])),
                    'confidence': float(i_ali[3]),
                    'annotation': annotation
                }
                frames_info.append(thisframe)

            videos[vname] = frames_info
    return videos


class DataGenerator(keras.utils.Sequence):
    def __init__(self, datadir, anndir, batch_size, n_seq_per_epoch, augmenter, sequence_len=16, split_video_len = 16, max_invalid=8, target_shape = (224, 224, 3), preprocessing='full', random_windows=False, dataset ='afew'):
        self.datadir = datadir
        self.anndir = anndir
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.n_seq_per_epoch = n_seq_per_epoch
        self.sequence_len = sequence_len
        self.max_invalid = max_invalid
        self.augmenter = augmenter
        self.preprocessing = preprocessing
        self.split_len = split_video_len

        if not random_windows:
            self.n_seq_per_epoch = self.batch_size * 2

        assert self.n_seq_per_epoch % batch_size == 0
        if dataset == 'affwild':
            self.dataset = 'affwild'
            self.videos = loadAffWild(datadir, anndir)
        elif dataset == 'afew':
            self.videos = loadAFEW(datadir)
            self.dataset = 'afew'

        else:
            raise

        self.random_windows = random_windows

        if not random_windows:

            splitted_videos_before_removing = self._split_videos()
            #print("Num seq before removing: ",self.count_seq(splitted_videos_before_removing))
            splitted_videos_after_removing = self._remove_invalid_sequences(splitted_videos_before_removing)
            #print("Num seq after removing: ",self.count_seq(splitted_videos_after_removing))
            partition = '' #todo handle
            splitted_videos_tuple = []
            if sequence_len == 1:
                for k,v in splitted_videos_after_removing.items():
                    for i in range(n_seq_per_epoch):
                        sequence_index = i % len(v)
                        splitted_videos_tuple.append((k,v[sequence_index]))
                #list in case of non-randomness


            else:
                for k,v in splitted_videos_after_removing.items():
                    for sequence in v:
                        splitted_videos_tuple.append((k,sequence))
                #list in case of non-randomness
            self.splitted_videos_tuple = splitted_videos_tuple

        self.on_epoch_end()
    def count_seq(self,list_map):
        counter = 0
        for k,v in list_map.items():
            counter += len(v)
        return counter
    def _read_frame(self, video_key, frame_num, label = ''):
        impath = os.path.join(self.datadir, label, video_key + '_aligned', "frame_det_00_%06d.%s" % (frame_num+1, IMAGE_EXT))
        frame = cv2.imread(impath)

        if frame is None:
            ##print("Unable to load %s" % impath)
            return None
        else:
            roi = DEFAULT_ROI
            # rotate and skew on bigger image avoiding black margin
            frame = self.augmenter.before_cut(frame,roi)
            aug_roi = self.augmenter.augment_roi(roi)
            frame = cut(frame,aug_roi)
            frame = self.augmenter.after_cut(frame)

        if self.preprocessing == 'full':
            frame = preprocessing_full(frame)
        elif self.preprocessing == 'imagenet':
            frame = preprocessing_imagenet(frame)
        elif self.preprocessing == 'vggface':
            frame = preprocessing_vggface2(frame)
        elif self.preprocessing == 'no':
            frame = preprocessing_no(frame)
        else:
            raise

        if frame.shape != self.target_shape:
            frame = cv2.resize(frame, (self.target_shape[1], self.target_shape[0]))

        return frame

    def _get_sequence(self, video_key, framerange):
        frames = []
        labels = []
        vinfo = self.videos[video_key]
        label = vinfo[0]['annotation']
        if type(label) is tuple:
            label = ''

        # make the augmentation steady for a whole sequence
        self.augmenter.reset_state()
        for thisframe in framerange:
            frames.append(self._read_frame(video_key, thisframe,label))
            labels.append(vinfo[thisframe]['annotation'])
        # Assign a neightboring not none frame to None frames
        not_none = next((f for f in frames if f is not None), None)
        if not_none is None:
            #print("all frames are none: ",video_key)
            #print(video_key,framerange)
            # All frames are None
            return None, labels

        for i in range(len(frames)):
            if frames[i] is None:
                frames[i] = not_none
            else:
                not_none = frames[i]
        len_middle = int((len(labels)/2))-1
        middle_label = labels[len_middle]

        if self.dataset == 'afew':
            label_encoder = LabelEncoder()
            hed_labels = to_categorical(label_encoder.fit_transform(CLASSES))
            middle_label = hed_labels[CLASSES.index(middle_label)]


        return np.array(frames), middle_label

    def _get_sequence_rand(self, video_key):

        assert self.max_invalid < self.sequence_len
        vinfo = self.videos[video_key]
        n = len(vinfo)
        for t in range(MAX_TRIES):

            startframe = random.randint(0, n - self.sequence_len - 1)
            #this solution works also for single images, in this case set sequence_len = 1 for ex.
            framerange = range(startframe, startframe + self.sequence_len)

            invalid = np.zeros((self.sequence_len,), dtype=np.uint8)
            for i, f in enumerate(framerange):

                vinfo = self.videos[video_key]
                label = vinfo[0]['annotation']
                if type(label) is tuple:
                    label = ''

                if vinfo[f]['annotation'][0] == -5.0 or vinfo[f]['annotation'][1] == -5.0:
                    invalid = np.ones((self.sequence_len,), dtype=np.uint8)
                    break



                frame_path_possible_problem = os.path.join(self.datadir, label, video_key + '_aligned', "frame_det_00_%06d.%s" % (f+1, IMAGE_EXT))
                frame_read = cv2.imread(frame_path_possible_problem)

                if vinfo[f]['success'] == False or vinfo[f]['confidence'] < 0.2:
                    invalid[i] = 1
                elif vinfo[f]['success'] == True and frame_read is None:
                    invalid[i] = 1

            if np.sum(invalid) <= self.max_invalid:
                return self._get_sequence(video_key, framerange)
       
    def __len__(self):
        if self.random_windows:
            return self.n_seq_per_epoch // self.batch_size
        else:
            # #print(len(self.splitted_videos_tuple) // self.batch_size)
            return len(self.splitted_videos_tuple) // self.batch_size

    def on_epoch_end(self):
        if self.random_windows:
            self.video_keys_order = list(self.videos.keys()).copy()
            random.shuffle(self.video_keys_order)

    def __getitem__(self, index):
        batch = []
        index *= self.batch_size
        # collect a batch
        X= np.empty((self.batch_size,self.sequence_len,self.target_shape[0],self.target_shape[1], self.target_shape[2]))
        Y = np.empty((self.batch_size,2))
        if self.dataset == 'afew':
            Y = np.empty((self.batch_size, 7))
        for i in range(index, index + self.batch_size):
            if self.random_windows:
                i = i % len(self.videos)
                video_key = self.video_keys_order[i]
                im, lbl = self._get_sequence_rand(video_key)
            else:
                video_key = self.splitted_videos_tuple[i][0]
                start_frame = list(self.splitted_videos_tuple[i][1].keys())[0]
                end_frame = list(self.splitted_videos_tuple[i][1].keys())[self.sequence_len-1]
                assert end_frame - start_frame == self.sequence_len-1
                framerange = range(start_frame, start_frame + self.sequence_len)
                im, lbl = self._get_sequence(video_key, framerange)


            tpl_obj = (im,lbl)
            batch.append(tpl_obj)
        for i in range(len(batch)):
            X[i] = batch[i][0]
            Y[i] = batch[i][1]
        #X = np.squeeze(X,axis=1)
        return X,Y

    def _split_videos(self):
        videos_splitted = {}
        for k,v in self.videos.items():
            n_video_sequences = int(len(v)//self.split_len)
            frame_sequences = []
            for i in range(n_video_sequences):
                current_index = i*self.split_len
                slice_map ={}
                for j in range(current_index,current_index+self.split_len):
                    slice_map[j] = v[j]
                frame_sequences.append(slice_map)
            videos_splitted[k] = frame_sequences
        return videos_splitted

    def _remove_invalid_sequences(self, splitted_videos):
        assert self.max_invalid < self.sequence_len
        for k,v in splitted_videos.items():
            v_new = []
            for sequence in v:
                invalid = 0
                for nframe, info in sequence.items():
                    
                    if info['annotation'][0] == -5.0 or info['annotation'][1] == -5.0:
                        invalid = self.max_invalid + 100
                        break

                    if info['success'] == False or info['confidence'] < 0.2:
                        invalid += 1
                    
                        
                if invalid > self.max_invalid:                    
                    pass
                else:
                    v_new.append(sequence)
            splitted_videos[k] = v_new
        return splitted_videos
