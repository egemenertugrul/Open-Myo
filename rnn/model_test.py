import os

import numpy as np
import pandas as pd
import sklearn
from keras.models import load_model
from sklearn import preprocessing

from michidk_dataset import MICHIDK_SUBJECTS_COUNT, MICHIDK_GESTURE_SET
from michidk_dataset.parser import parse_recordings_as_dataset


def load_best_model():
    current_filepath = os.path.dirname(os.path.realpath(__file__))
    model_name = "S1-13_G3_r_1674823233-49-0.825.model"
    model_path = os.path.join(current_filepath, "models", model_name)
    model = load_model(model_path)
    return model

if __name__ == '__main__':
    model = load_best_model()

    subjects_filter = range(1, MICHIDK_SUBJECTS_COUNT + 1)  # range(1, 2)
    gestures_filter = MICHIDK_GESTURE_SET
    n_gestures = len(gestures_filter)
    arms_filter = ['r']

    n_steps, n_length, n_features = 4, 50, 8  # ops happen on 50

    # X_data, Y_data = parse_recordings_as_dataset(subjects_filter, gestures_filter, arms_filter)
    # X_data = X_data.reshape((X_data.shape[0], n_steps, n_length, n_features))
    #
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, Y_data, test_size=0.2,
    #                                                                             random_state=10, shuffle=True)

    # print("Evaluate on test data")
    # results = model.evaluate(X_test[s:s+3], y_test[s:s+3], batch_size=1)
    # print("test loss, test acc:", results)

    # --

    y = pd.read_csv("../michidk_dataset/s6_r_4/s6_r_4-paper-7-emg.csv", usecols=range(2, 10))
    y = np.array(y)
    selected_data = y[150:350]
    selected_data = selected_data.reshape(-1, 8, 200)
    selected_data_norm = np.interp(selected_data, (-128, 127), (-1, +1))
    selected_data_norm = selected_data_norm.reshape(1, 200, 8)
    X_test = selected_data_norm.reshape((selected_data_norm.shape[0], n_steps, n_length, n_features))

    s = 50

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    predictions = model.predict(X_test)
    print("predictions shape:", predictions.shape)

    gestures = ['rock', 'paper', 'scissors']  # , 'paper', 'scissors'
    lb = preprocessing.LabelBinarizer()

    print(predictions)
    lb.fit(gestures)
    pred_dec = lb.inverse_transform(predictions)
    print(pred_dec)
    # gt_dec = lb.inverse_transform(y_test[s:s+3])
    # print(pred_dec, 'vs', gt_dec)

    # incorrects = np.nonzero(model.predict_class(X_test).reshape((-1,)) != y_test)
    # print(incorrects)
