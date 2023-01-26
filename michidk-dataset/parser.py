import glob
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn.model_selection
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, Conv1D, MaxPooling1D, Flatten
# from keras.optimizers import Adam
from keras.optimizer_v2.adam import Adam
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from features import disjoint_segmentation_2, wl, overlapping_segmentation, overlapping_segmentation_2, rms, zc
from myo_zmq.recorder import Recording

rootdir = '.'

GESTURE_SET = ['rock', 'paper', 'scissors']
ARM_SET = ['l', 'r']
SUBJECTS_COUNT = 13

class Michidk_Recording(Recording):
    def __init__(self,  *args, gesture, take):
        super(Michidk_Recording, self).__init__(*args)
        self.gesture = gesture
        self.take = take

def parse_recordings(
        subjects_filter: range(1, SUBJECTS_COUNT + 1),
        gestures_filter=GESTURE_SET,
        arms_filter = ARM_SET,
        file_type_filter=('emg')):
    if subjects_filter.start < 1 or subjects_filter.stop > SUBJECTS_COUNT + 1:
        raise Exception("Error! Check subjects filter.")

    for gesture in gestures_filter:
        if gesture not in GESTURE_SET:
            raise Exception("Error! Check gestures filter.")

    recordings = []

    for subdir, dirs, files in os.walk(rootdir):
        if subdir == rootdir:
            continue

        strip_subdir = subdir.lstrip('.\\')
        subject_id, arm, sample_index = strip_subdir.split('_')
        subject_id = int(subject_id.lstrip('s'))

        if subject_id not in subjects_filter:
            continue
        if arm not in arms_filter:
            continue

        for file in files:
            # if file.endswith('emg.csv'):
            subject_info, gesture, take_index, file_type = file.split('-')
            take_index = int(take_index)
            file_type = file_type.rstrip('.csv')
            if gesture in gestures_filter:
                if file_type in file_type_filter:
                    print(os.path.join(subdir, file))
                    file_path = os.path.join(os.getcwd(), strip_subdir, file)
                    recording = Michidk_Recording(f"{subject_info}-{gesture}-{take_index}", file_path,
                                                  gesture=gesture, take=take_index)
                    recording.fill_from_file(usecols=range(2, 10))
                    recordings.append(recording)
    return recordings

def parse_recordings_as_dataset(subjects_filter, gestures_filter, arms_filter):
    recs: List[Michidk_Recording] = parse_recordings(
        subjects_filter= subjects_filter, # range(10,11), #
        gestures_filter=gestures_filter,
        arms_filter=arms_filter
    )

    segment_length = 52
    skip = 26
    channels = 8

    X_data = []
    Y_data = []
    for rec in recs:
        recording_data = rec.get_data()
        # large_chunks = disjoint_segmentation_2(recording_data, 200).reshape(-1, 8, 200)
        s = 150
        r = 200

        large_chunks = recording_data[s:s+r]
        large_chunks = large_chunks.reshape(-1, 8, r)
        for chunk in large_chunks:
            # overlapping_segmentation_fn = partial(overlapping_segmentation_2, n_samples=segment_length, skip=skip)
            # segments = np.array(list(map(overlapping_segmentation_fn, chunk)))  # .reshape(-1, segment_length, channels)
            #
            # wl_applied = np.apply_along_axis(wl, axis=2, arr=segments).reshape(-1, channels)
            # rms_applied = np.apply_along_axis(rms, axis=2, arr=segments).reshape(-1, channels)
            # zc_applied = np.apply_along_axis(zc, axis=2, arr=segments).reshape(-1, channels)
            # #
            # wl_applied = np.divide(wl_applied, 13008)
            # rms_applied = np.divide(rms_applied, 128)
            #
            # x_data = np.concatenate((wl_applied, rms_applied, zc_applied), axis=1)

            # X_data.append(rms_applied)

            x_data = chunk.reshape(-1, channels)
            x_data = np.interp(x_data, (-128, 127), (-1, +1))
            X_data.append(x_data)

            # X_data.extend(segments)
        Y_data.extend([rec.gesture] * len(large_chunks))

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # values_to_feed_to_scaler = X_data.reshape(X_data.shape[0] * X_data.shape[1] * X_data.shape[2], 1)
    # scaler = StandardScaler(with_std=True, with_mean=True) # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(values_to_feed_to_scaler)
    # print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, math.sqrt(scaler.var_)))
    # X_data_normalized = scaler.transform(values_to_feed_to_scaler)
    # X_data = X_data_normalized.reshape(X_data.shape[0], X_data.shape[1], X_data.shape[2])

    lb = preprocessing.LabelBinarizer()
    Y_data_enc = lb.fit_transform(Y_data)

    return X_data, Y_data_enc

    # print(len(X_data), len(Y_data_enc))

if __name__ == '__main__':
    subjects = range(1, SUBJECTS_COUNT + 1)  # range(1, 2)
    gestures = ['rock', 'paper', 'scissors']  # , 'paper', 'scissors'
    arms = ['r']

    n_steps, n_length, n_features = 4, 50, 8  # ops happen on 50
    # n_steps, n_length, n_features = 3, 6, 8

    X_data, Y_data = parse_recordings_as_dataset(subjects, gestures, arms)
    X_data = X_data.reshape((X_data.shape[0], n_steps, n_length, n_features))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, Y_data, test_size=0.2,
                                                                                random_state=10, shuffle=True)


    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    # input_shape=X_train.shape[1:],
    model.add(LSTM(100, return_sequences=False, recurrent_dropout=0.35, dropout=0.05, activation='tanh'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    activation_fn = 'sigmoid' if y_train.shape[1] == 1 else 'softmax'
    model.add(Dense(y_train.shape[1], activation=activation_fn))

    # Keras optimizer defaults:
    # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
    # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
    # SGD    : lr=0.01,  momentum=0.,                             decay=0.

    learning_rate = 0.001
    opt_fn = Adam
    opt = opt_fn(lr=learning_rate, decay=1e-6)

    loss_fn = 'binary_crossentropy' if y_train.shape[1] == 1 else 'categorical_crossentropy'

    model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])
    model.summary()
    # make predictions
    # trainPredict = model.predict(X_train)
    # testPredict = model.predict(X_test)

    NAME = f"S{subjects.start}-{subjects.stop-1}_G{len(gestures)}_{'-'.join([a for a in arms])}_{int(time.time())}"

    tensorboard = TensorBoard(log_dir="../logs/{}".format(NAME))

    filepath = NAME + "-{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("../models/{}.model".format(filepath, monitor='val_acc', verbose=1,
                                                             save_best_only=True, mode='auto'))

    history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=2,
                        validation_data=(X_test, y_test),
                        callbacks=[tensorboard, checkpoint]
                        )

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("../models/{}".format(NAME))

    # print(len(X_train), len(X_test), len(y_train), len(y_test))







# y = pd.read_csv("recordings/palm_open_0.csv", usecols = signal_range)
