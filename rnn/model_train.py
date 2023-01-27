from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from michidk_dataset import MICHIDK_SUBJECTS_COUNT, MICHIDK_GESTURE_SET
from michidk_dataset.parser import parse_recordings_as_dataset

import time
import sklearn.model_selection
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from keras.optimizer_v2.adam import Adam

from rnn.GridCV_scores_printer import save_gridcv_scores

#conv_dropout_rate=.5, regular_dropout_rate=.2, recurrent_dropout_rate=.35
def create_model(input_shape, output_shape, conv_dropout_rate, regular_dropout_rate, recurrent_dropout_rate):
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(conv_dropout_rate)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100, return_sequences=False, recurrent_dropout=recurrent_dropout_rate, dropout=regular_dropout_rate, activation='tanh'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(regular_dropout_rate))
    activation_fn = 'sigmoid' if output_shape == 1 else 'softmax'
    model.add(Dense(output_shape, activation=activation_fn))
    # Keras optimizer defaults:
    # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
    # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
    # SGD    : lr=0.01,  momentum=0.,                             decay=0.
    learning_rate = 0.001
    opt_fn = Adam
    opt = opt_fn(lr=learning_rate, decay=1e-6)
    loss_fn = 'binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy'
    model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])
    # model.summary()
    return model


if __name__ == '__main__':
    subjects_filter = range(1, MICHIDK_SUBJECTS_COUNT + 1)  # range(1, 2)
    gestures_filter = MICHIDK_GESTURE_SET
    n_gestures = len(gestures_filter)
    arms_filter = ['r']

    NAME = f"S{subjects_filter.start}-{subjects_filter.stop - 1}_G{len(gestures_filter)}_{'-'.join([a for a in arms_filter])}_{int(time.time())}"

    n_steps, n_length, n_features = 4, 50, 8  # ops happen on 50
    # n_steps, n_length, n_features = 3, 6, 8

    X_data, Y_data = parse_recordings_as_dataset(subjects_filter, gestures_filter, arms_filter)
    X_data = X_data.reshape((X_data.shape[0], n_steps, n_length, n_features))

    input_shape = (None, n_length, n_features)
    output_shape = n_gestures

    isGridCV = True

    if not isGridCV:
        epochs = 50
        batch_size = 12

        test_size = 0.2
        isShuffle = True

        isTensorboard = True
        isCheckpoint = True
        isEarlystopping = False

        callbacks = []

        if isTensorboard:
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            callbacks.append(tensorboard)

        if isCheckpoint:
            filepath = NAME + "-{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
            checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1,
                                                                     save_best_only=True, mode='auto'))
            callbacks.append(checkpoint)

        if isEarlystopping:
            earlystopping = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=3,
                verbose=1,
                mode='auto',
                baseline=None,
                restore_best_weights=False
            )
            callbacks.append(earlystopping)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, Y_data, test_size=test_size,
                                                                                    random_state=10, shuffle=isShuffle)
        model = create_model(
            input_shape=input_shape,
            output_shape=output_shape,
            conv_dropout_rate=.5,
            regular_dropout_rate=.2,
            recurrent_dropout_rate=.35
        )

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks
                            )

        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save("models/{}".format(NAME))
    else:
        model = KerasClassifier(build_fn=create_model, input_shape=input_shape, output_shape=output_shape)
        epochs = [30]  # [10, 30, 50, 70, 100]
        batch_size = [12]  # [12, 24, 32, 48]
        conv_dropout_rate = [0.25, 0.5, 0.75]
        regular_dropout_rate = [0.25, 0.5, 0.75]
        recurrent_dropout_rate = [0.25, 0.5, 0.75]
        param_grid = dict(
            epochs=epochs,
            batch_size=batch_size,
            conv_dropout_rate=conv_dropout_rate,
            regular_dropout_rate=regular_dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate
        )
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=3, verbose=2)
        grid_result = grid.fit(X_data, Y_data)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        save_gridcv_scores(grid, f"gridcv_results-{NAME}.csv")
