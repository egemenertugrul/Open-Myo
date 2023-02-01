from functools import partial

from features import disjoint_segmentation_2, wl, rms, zc, overlapping_segmentation_2
from michidk_dataset import MICHIDK_SUBJECTS_COUNT, MICHIDK_GESTURE_SET

import os
from typing import List
import numpy as np
from sklearn import preprocessing

from myo_zmq.recorder import Recording
from filters import filter_signal, rectify_signal, envelope_signal

rootdir = os.path.dirname(os.path.realpath(__file__))

ARM_SET = ['l', 'r']


class Michidk_Recording(Recording):
    def __init__(self, *args, gesture, take):
        super(Michidk_Recording, self).__init__(*args)
        self.gesture = gesture
        self.take = take


def parse_recordings(
        subjects_filter: range(1, MICHIDK_SUBJECTS_COUNT + 1),
        gestures_filter=MICHIDK_GESTURE_SET,
        arms_filter=ARM_SET,
        file_type_filter=('emg')):
    if subjects_filter.start < 1 or subjects_filter.stop > MICHIDK_SUBJECTS_COUNT + 1:
        raise Exception("Error! Check subjects filter.")

    for gesture in gestures_filter:
        if gesture not in MICHIDK_GESTURE_SET:
            raise Exception("Error! Check gestures filter.")

    recordings = []

    for subdir, dirs, files in os.walk(rootdir):
        if subdir == rootdir:
            continue

        subdir_basename = os.path.basename(os.path.normpath(subdir))
        try:
            subject_id, arm, sample_index = subdir_basename.split('_')
        except:
            continue

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
                    file_path = os.path.join(subdir, file)
                    recording = Michidk_Recording(f"{subject_info}-{gesture}-{take_index}", file_path,
                                                  gesture=gesture, take=take_index)
                    recording.fill_from_file(usecols=range(2, 10))
                    recordings.append(recording)
    return recordings


def parse_recordings_as_dataset(subjects_filter, gestures_filter, arms_filter,
                                postprocess=dict({'filter': True, 'rectify': True, 'envelope': False})):
    recs: List[Michidk_Recording] = parse_recordings(
        subjects_filter=subjects_filter,  # range(10,11), #
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
        s = 150
        r = 400

        large_chunks = disjoint_segmentation_2(recording_data, r).reshape(-1, 8, r)
        # large_chunks = recording_data[s:s + r]
        # large_chunks = large_chunks.reshape(-1, 8, r)
        x = []
        for chunk in large_chunks:
            x_data = chunk

            x_data = np.interp(x_data, (-128, 127), (-1, +1))
            if np.any(postprocess.values()):
                if postprocess['filter']:
                    x_data = np.apply_along_axis(func1d=filter_signal, axis=len(x_data.shape) - 1, arr=x_data, order=1,
                                                 sfreq=200, high_band=10,
                                                 low_band=80)
                if postprocess['rectify']:
                    x_data = rectify_signal(x_data)

                if postprocess['envelope']:
                    x_data = np.apply_along_axis(func1d=envelope_signal, axis=len(x_data.shape) - 1, arr=x_data,
                                                 order=5,
                                                 low_pass=80, sfreq=200)

            # x_data = x_data.reshape(-1, channels)
            # X_data.append(x_data)

            overlapping_segmentation_fn = partial(overlapping_segmentation_2, n_samples=segment_length, skip=skip)
            segments = np.array(
                list(map(overlapping_segmentation_fn, x_data)))# .reshape(-1, channels) # .reshape(-1, segment_length, channels)

            # wl_applied = np.apply_along_axis(wl, axis=2, arr=segments).reshape(-1, channels)
            # rms_applied = np.apply_along_axis(rms, axis=2, arr=segments).reshape(-1, channels)
            # zc_applied = np.apply_along_axis(zc, axis=2, arr=segments).reshape(-1, channels)
            #
            # wl_applied = np.divide(wl_applied, 13008)
            # rms_applied = np.divide(rms_applied, 128)

            # x_data = np.concatenate((wl_applied, rms_applied, zc_applied), axis=1)

            # X_data.append(wl_applied)

            x.extend(segments)
        X_data.append(x)
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
