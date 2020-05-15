import os
from Dataset.Dataset_Utils.old_files_afew.dataset_tools import split_video, list_dirs
import numpy as np
import statistics
from collections import Counter
from itertools import groupby
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
# from Models.seresnet50 import SEResNet50
# from Models.lstm import LSTM_net
# import Models.Config.SEResNet50_config as config
# import Models.Config.LSTM_config as lstm_config
# from keras.models import Model, Sequential
# from keras.layers import Dense, TimeDistributed
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import pearsonr
import math

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class Inference:
    """
    baseline_pretrained_on can be: 
        'AFEW',
        'FER'
     """

    def __init__(self, model=None, partition='Val', baseline_pretrained_on='AFEW', custom_inference=False, time_step=16,
                 classification=True):
        print('Init Inference')
        self.partition = partition
        self.model = model
        self.pretrain_data = baseline_pretrained_on
        self.classes = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
        self.custom_inference = custom_inference
        # predicted_final contains the prediction for the whole clip
        self.predicted = []
        # it contains the true labels for each clip
        self.true = []
        self.slice_len = time_step
        self.classification = classification

    def get_stats(self, true, predicted):
        if self.classification:
            classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
            stats = []
            cm = confusion_matrix(true, predicted, classes)
            stats.append(cm)
            cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
            stats.append(cm)
            stats.append(accuracy_score(true, predicted))
            stats.append(classification_report(self.true, self.predicted))
            self.true.clear()
            self.predicted.clear()
        else:
            true = np.squeeze(np.array(true))
            predicted = np.squeeze(np.array(predicted))
            stats = self._calculate_ccc_index(true, predicted)

        return stats

    def predict_generator(self, gen=None, mode=''):
        print('Evaluating ', str(len(gen)), ' batches')
        for i, batch in enumerate(tqdm(gen)):
            self.predict_on_batch(batch, mode)

        return self.true, self.predicted

    def predict_on_batch(self, batch, mode):
        for x, y in zip(batch[0], batch[1]):
            if self.classification:
                self.true.append(self.classes[np.argmax(y)])

                if self.custom_inference:
                    if mode == 'stride1':
                        pred = self.custom_prediction_stride1(x, y)
                    elif mode == 'overlap':
                        pred = self.custom_prediction_overlapp(x, y)
                    else:
                        pred = self.custom_prediction(x, y)
                    self.predicted.append(pred)
                else:
                    x = x[np.newaxis, ...]
                    prediction = self.model.predict(x)
                    self.predicted.append(self.classes[np.argmax(prediction)])
            # else regression task
            else:
                # x are correct frames of a video and y a list of tuples (val,arousal)
                # arousal and valence values
                self.true.append(y)
                pred = self._regression_prediction(x)
                # pred is a list of predictions containing list of (arousal, valence) values
                self.predicted.append(pred)

    def custom_prediction(self, x, y):
        # prediction_frames_classes stores predicted label (the one with highest probability) for each frame of the clip
        predictions_frames_classes = []
        # we need prediction_single_clip_prob since it could be happen that the mode outputs more than one label, so the
        # idea is to store all the frames prediction in order to take the label with the max of the probabilities sum
        # over the frames
        predictions_single_clip_prob = []

        # evaluate batch
        # check if model requires sequences or single frame (c3d/cnn+rnn or only cnn)
        if len(self.model.layers[0].input_shape) > 4:
            # slices
            item = {
                'frames': x,
                'label': y,
            }
            splitted = split_video(item=item, split_len=self.slice_len, partition=self.partition)
            for video in splitted:
                item = video['frames']
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                predictions_single_clip_prob.append(prediction)
                predicted_class_frame = np.argmax(prediction)
                predictions_frames_classes.append(predicted_class_frame)
        else:
            # single_frame
            for item in x:
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                # in case we choose ferplus weights, we have eight outputs, so we decide to not consider the last output
                if self.pretrain_data == 'FER':
                    prediction = prediction[0][:-1]
                    prediction = self._from_fer_to_afew_pred(prediction)
                predictions_single_clip_prob.append(prediction)
                predicted_class_frame = np.argmax(prediction)
                predictions_frames_classes.append(predicted_class_frame)
        try:
            mode_prediction = statistics.mode(predictions_frames_classes)
            mode_prediction = self.classes[mode_prediction]
        except statistics.StatisticsError as e:
            # print("eccezione")
            freqs = groupby(Counter(predictions_frames_classes).most_common(), lambda x: x[1])
            multiple = [val for val, count in next(freqs)[1]]
            # print("multiple ",multiple)
            average_prediction = np.sum(predictions_single_clip_prob, axis=0)
            if self.pretrain_data == 'FER':
                average_prediction = np.squeeze(average_prediction)
            else:
                average_prediction = average_prediction[0]
            average_winners = []
            for i in range(0, len(multiple)):
                average_winners.append(average_prediction[multiple[i]])
            winner = multiple[np.argmax(average_winners)]
            mode_prediction = self.classes[winner]

        return mode_prediction

    def custom_prediction_stride1(self, x, y):
        # predictions_frames_classes stores the predicted label (the one with the highest probability) for each frame
        # of the clip
        predictions_frames_classes = []
        # we need prediction_single_clip_prob since it could be happen that the mode outputs more than one label
        # so the idea is to store all the frames prediction in order to take the label with the maximum of the
        # probabilities sum over the frames
        predictions_single_clip_prob = []

        # evaluate batch
        # check if model requires sequences or single frame (c3d/cnn+rnn or only cnn)
        if len(self.model.layers[0].input_shape) > 4:
            # slices
            if len(x) < self.slice_len:
                item = {
                    'frames': x,
                    'label': y,
                }
                x = split_video(item=item, split_len=self.slice_len, partition=self.partition)[0]['frames']
            for i in range((len(x) - self.slice_len) + 1):
                # item = video['frames']
                item = x[i:i + self.slice_len]
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                predictions_single_clip_prob.append(prediction)
                predicted_class_frame = np.argmax(prediction)
                predictions_frames_classes.append(predicted_class_frame)
        else:
            # single_frame
            # for item in (x):
            if len(x) < self.slice_len:
                index = len(x) - 1
            else:
                index = self.slice_len - 1
            for i in range(index, len(x)):
                item = x[i]
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                # in case we choose ferplus weights, in that case we have eight outputs, so we decide to not consider
                # the last output (mapping)
                if self.pretrain_data == 'FER':
                    prediction = prediction[0][:-1]
                    prediction = self._from_fer_to_afew_pred(prediction)
                predictions_single_clip_prob.append(prediction)
                predicted_class_frame = np.argmax(prediction)
                predictions_frames_classes.append(predicted_class_frame)
        try:
            mode_prediction = statistics.mode(predictions_frames_classes)
            mode_prediction = self.classes[mode_prediction]
        except statistics.StatisticsError as e:
            # print("eccezione")
            freqs = groupby(Counter(predictions_frames_classes).most_common(), lambda x: x[1])
            multiple = [val for val, count in next(freqs)[1]]
            # print("multiple ",multiple)
            average_prediction = np.sum(predictions_single_clip_prob, axis=0)
            if self.pretrain_data == 'FER':
                average_prediction = np.squeeze(average_prediction)
            else:
                average_prediction = average_prediction[0]
            average_winners = []
            for i in range(0, len(multiple)):
                average_winners.append(average_prediction[multiple[i]])
            winner = multiple[np.argmax(average_winners)]
            mode_prediction = self.classes[winner]

        return mode_prediction

    def custom_prediction_overlapp(self, x, y):
        # predictions_frames_classes stores the predicted label (the one with the highest probability) for each frame
        # of the clip
        predictions_frames_classes = []
        # we need prediction_single_clip_prob since it could be happen that the mode outputs more than one label
        # so the idea is to store all the frames prediction in order to take the label with the maximum of the
        # probabilities sum over the frames
        predictions_single_clip_prob = []

        # evaluate batch
        # check if model requires sequences or single frame (c3d/cnn+rnn or only cnn)
        if len(self.model.layers[0].input_shape) > 4:
            # slices
            if len(x) < self.slice_len:
                item = {
                    'frames': x,
                    'label': y,
                }
                x = split_video(item=item, split_len=self.slice_len, partition=self.partition)[0]['frames']
            for i in range(0, (len(x) - self.slice_len + 1), int(self.slice_len / 2)):
                # item = video['frames']
                item = x[i:i + self.slice_len]
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                predictions_single_clip_prob.append(prediction)
                predicted_class_frame = np.argmax(prediction)
                predictions_frames_classes.append(predicted_class_frame)
        else:
            # single_frame
            # for item in (x):
            if len(x) < self.slice_len:
                index = len(x) - 1
            else:
                index = self.slice_len - 1
            for i in range(index, len(x)):
                item = x[i]
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                # in case we choose ferplus weights, in that case we have eight outputs, so we decide to not consider
                # the last output (mapping)
                if self.pretrain_data == 'FER':
                    prediction = prediction[0][:-1]
                    prediction = self._from_fer_to_afew_pred(prediction)
                predictions_single_clip_prob.append(prediction)
                predicted_class_frame = np.argmax(prediction)
                predictions_frames_classes.append(predicted_class_frame)
        try:
            mode_prediction = statistics.mode(predictions_frames_classes)
            mode_prediction = self.classes[mode_prediction]
        except statistics.StatisticsError as e:
            # print("eccezione")
            freqs = groupby(Counter(predictions_frames_classes).most_common(), lambda x: x[1])
            multiple = [val for val, count in next(freqs)[1]]
            # print("multiple ",multiple)
            average_prediction = np.sum(predictions_single_clip_prob, axis=0)
            if self.pretrain_data == 'FER':
                average_prediction = np.squeeze(average_prediction)
            else:
                average_prediction = average_prediction[0]
            average_winners = []
            for i in range(0, len(multiple)):
                average_winners.append(average_prediction[multiple[i]])
            winner = multiple[np.argmax(average_winners)]
            mode_prediction = self.classes[winner]

        return mode_prediction

    def _from_fer_to_afew_pred(self, prediction):
        return np.array([prediction[4], prediction[5], prediction[6], prediction[1], prediction[0], prediction[3],
                          prediction[2]])

    def _regression_prediction(self, x):
        predictions = []

        # multi-frame model
        if len(self.model.layers[0].input_shape) > 4:
            for i in tqdm(range((len(x) - self.slice_len) + 1)):
                # item = video['frames']
                item = x[i:i + self.slice_len]
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)
                predictions.append(prediction)

            starting_index_remain = (len(x) - self.slice_len) + 1
            for i in range(starting_index_remain, len(x)):
                predictions.append(prediction)

        # single-frame model
        else:
            for i in tqdm(range(len(x))):
                item = x[i]
                item = item[np.newaxis, ...]
                prediction = self.model.predict(item)  # arousal valence prediction
                predictions.append(prediction)

        return predictions

    # true and predicted are list of list (one list for each video)
    def _calculate_ccc_index(self, true, predicted):

        ccc_arousal_list = []
        ccc_valence_list = []
        true = np.array(true)
        # print(predicted.shape)
        # print(np.array(predicted[0]).shape)
        # predicted = np.array(predicted)
        for i in range(len(true)):
            # valence
            true_valence = true[i][:, 0]
            predicted_valence = np.squeeze(np.array(predicted[i]))[:, 0]
            ccc_valence = self._calculate_ccc_single(true_valence, predicted_valence)
            ccc_valence_list.append(ccc_valence)

            # arousal
            true_arousal = true[i][:, 1]
            predicted_arousal = np.squeeze(np.array(predicted[i]))[:, 1]
            ccc_arousal = self._calculate_ccc_single(true_arousal, predicted_arousal)
            ccc_arousal_list.append(ccc_arousal)

            print("ccc (v,a): {} | {}".format(ccc_valence, ccc_arousal))

        ccc_arousal_mean = np.mean(ccc_arousal_list)
        ccc_valence_mean = np.mean(ccc_valence_list)

        return ccc_valence_mean, ccc_arousal_mean

    def _calculate_ccc_single(self, true, predicted):
        new_true = []
        new_pred = []

        assert len(true) == len(predicted)
        for i in range(len(true)):
            if true[i] != -5.0:
                new_true.append(true[i])
                new_pred.append(predicted[i])

        mu_true = np.mean(new_true)
        mu_predicted = np.mean(new_pred)
        pearson_coeff = pearsonr(new_true, new_pred)[0]
        if math.isnan(pearson_coeff):
            return 0.0
        sigma_true_square = np.var(new_true)
        sigma_predicted_square = np.var(new_pred)
        ccc = (2 * pearson_coeff * math.sqrt(sigma_predicted_square) * math.sqrt(sigma_true_square)) / (
                (sigma_predicted_square + sigma_true_square) + (mu_predicted - mu_true) ** 2)

        return ccc


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)


def stats_to_csv(stats_path='', only_cnn=False):
    import csv
    name_csv = os.path.join(stats_path, os.path.basename(stats_path) + '.csv')
    with open(name_csv, 'a') as csvFile:
        writer = csv.writer(csvFile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if only_cnn:
            param = ['Date', '# of frames for casting', 'Initial learning rate', 'Learning rate decay factor',
                     'Learning rate decay epoch', 'Epochs', 'Batch size', 'Accuracy full video [slices no-overlapped]',
                     'Accuracy full video [slices overlap of ts/2]',
                     'Accuracy full video [slices shifted of 1 frame at time]']
        else:
            param = ['Date', 'Time step', 'Initial learning rate', 'Learning rate decay factor',
                     'Learning rate decay epoch', 'Epochs', 'Batch size', 'Layers', 'Dropouts', 'Bidirectional',
                     'Accuracy full video [slices no-overlapped]', 'Accuracy full video [slices overlap of ts/2]',
                     'Accuracy full video [slices shifted of 1 frame at time]']
        writer.writerow(param)
        csvFile.close()
    error_c = 0
    # iterate partition
    for experiment in sorted(list_dirs(stats_path)):
        # experiment is the folder containg all params on train (folder name) and all stats in stats.log
        param = os.path.basename(experiment).split('-')
        params = []
        keys = []
        i = 0
        params.append(param[0])
        while i < len(param):
            try:
                if param[i] == 'no':
                    params.append('no')
                if param[i] == 'bd':
                    params.append('bd')
                else:
                    if only_cnn and i == 2:
                        params.append(str(param[i]))
                    else:
                        n = float(param[i])
                        params.append(str(n))
            except:
                keys.append(param[i])
            i += 1

        i = 0

        stats_log_path = os.path.join(experiment, 'stats.log')
        tot_acc = []

        try:
            with open(stats_log_path) as stats_file:
                for line in stats_file:
                    if 'Accuracy Score : ' in line:
                        tot_acc.append(str(round(float(line.split('Accuracy Score : ')[1]), 3)))
            if only_cnn:
                params += tot_acc[-1:]
                params += ['none', 'none']
            else:
                params += tot_acc[-3:]

            while i < (len(params) - 3):
                if params[i] != 'no':
                    if params[i] == params[i + 1] == params[i + 2]:
                        params[i] = params[i] + '-' + params[i + 1] + '-' + params[i + 2]
                        del params[i + 2]
                        del params[i + 1]
                    if params[i] == params[i + 1]:
                        params[i] = params[i] + '-' + params[i + 1]
                        del params[i + 1]
                i += 1
            with open(name_csv, 'a') as csvFile:
                writer = csv.writer(csvFile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(params)
                csvFile.close()
        except:
            error_c += 1
            print('error on stats.log     |     error count: ', str(error_c))


# TEST INFERENCE -------------------

if __name__ == '__main__':
    """

    #NET
    INPUT_SHAPE = config.INPUT_SHAPE
    
    # MODEL ----------------------
    plus_rnn = True
    #instanciate cnn+lstm model
    model = Sequential()
    if plus_rnn:
        seres50 = SEResNet50(input_shape=INPUT_SHAPE, classes=NUM_CLASSES, lpf_size=config.LPF_SIZE,weights='AFEW',include_top=False)
        #wrap seres50 on TimeDistribuited and add in new model
        model.add(TimeDistributed(seres50,input_shape=(lstm_config.TIME_STEP, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])))
        #add recurrent network
        lstm = LSTM_net(input_shape=(lstm_config.TIME_STEP,lstm_config.FEATURES_DIM),output_dim = NUM_CLASSES,layers_size=lstm_config.LAYERS_SIZE,layers_dropout=lstm_config.LAYERS_DROPOUT, bidirectional = lstm_config.BIDIRECTIONAL,batch_norm = lstm_config.BATCH_NORM)
        model.add(lstm.get_model())
        #model.load_weights(lstm_config.SENET50_LSTM_AFEW_WEIGHTS_PATH)
        model.load_weights('/data/s4175719/AllTrainHistories/Random_Val/seres_lstm/afew/20_Nov_2019_16_51-lr-0.01-lrdf-0.1-lrde-20-ep-120-b-12-l-128-128-d-0.7-0.7-bd/checkpoint.117.hdf5')

    else:
        seres50 = SEResNet50(input_shape=INPUT_SHAPE, classes=NUM_CLASSES, lpf_size=config.LPF_SIZE,weights='AFEW',include_top=True)
        model=seres50

    model.summary()

    #DATASET --------------------
    custom=True

    if custom and not plus_rnn:
        dataset_validation = AFEW_Dataset(partition='Val', target_shape=config.INPUT_SHAPE, augment=False)
        gen=dataset_validation.get_video_generator(custom_inference = True)
    elif not custom and not plus_rnn:
        dataset_validation = AFEW_Dataset(partition='Val', target_shape=config.INPUT_SHAPE, augment=False,cast_to_imgs= True)
        gen=dataset_validation.get_generator()
    elif custom and plus_rnn:
        dataset_validation = AFEW_Dataset(partition='Val', target_shape=config.INPUT_SHAPE, augment=False)
        gen=dataset_validation.get_video_generator(custom_inference = True)
    elif not custom and plus_rnn:
        dataset_validation = AFEW_Dataset(partition='Val', target_shape=config.INPUT_SHAPE, augment=False,split = True)
        gen=dataset_validation.get_video_generator(random_overlapped = False,custom_inference = False)

    #INFERENCE -------------------
    pre_train_on = 'AFEW'
    if custom:
        inference = Inference(model = model, custom_inference = True,baseline_pretrained_on=pre_train_on)
    else:
        inference = Inference(model = model, custom_inference = False,baseline_pretrained_on=pre_train_on)

    (true,pred)=inference.predict_generator(gen)
    results = inference.get_stats()

    #RESULTS ---------------------

    print('Confusion Matrix :')
    print(results[0]) 
    print('Accuracy Score :',results[1]) 
    print('Report : ')
    print(results[2]) """
    stats_to_csv(stats_path='/data/s4175719/AllTrainHistories/seres_lstm/afew/2_livelli_wm_and_reg_last',
                 only_cnn=False)
