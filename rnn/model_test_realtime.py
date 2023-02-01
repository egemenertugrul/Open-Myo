import ctypes
import multiprocessing
import time
from multiprocessing import Process, Queue

import numpy as np
from sklearn import preprocessing

from rnn.model_test import load_best_model
from subscriber import MainLoopThread

import pygame

# def shared_array(shape):
#     """
#     Form a shared memory numpy array.
#
#     http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
#     """
#
#     shared_array_base = multiprocessing.Array(ctypes.c_double, shape[0] * shape[1])
#     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#     shared_array = shared_array.reshape(*shape)
#     return shared_array

r = 100

def transform_data_for_inference(data):
    _data = np.array(data)
    _data = _data.reshape(-1, 8, r)
    _data_norm = np.interp(_data, (-128, 127), (-1, +1))
    _data_norm = _data_norm.reshape(1, r, 8)
    # n_steps, n_length, n_features = 4, 50, 8  # ops happen on 50
    # _data_norm = _data_norm.reshape((_data_norm.shape[0], n_steps, n_length, n_features))
    return _data_norm # .tolist()


def model_test_realtime(emgQueue: Queue, inference_buffer: Queue):
    buffer = []
    buffer_len = r

    def process_data(epoch_time, emg_data):
        nonlocal inference_buffer
        buffer.append(emg_data)

        if len(buffer) > buffer_len:
            buffer.pop(0)

        if len(buffer) == buffer_len:
            inference_buffer.put(buffer)

    while True:
        # process_data(0, [1] * 8)

        if emgQueue.empty():
            continue

        data = emgQueue.get()

        shape = np.array(data['emg']).shape
        epoch_time = float(data['time'])

        if shape[0] == 2:
            for i in range(len(data['emg'])):
                emg_data = data['emg'][i]
                emg_data = list(map(float, emg_data))
                process_data(epoch_time, emg_data)
        else:
            emg_data = list(map(float, data['emg']))
            process_data(epoch_time, emg_data)

def realtime_loop(inference_buffer: multiprocessing.Queue):
    def sleep(duration_secs):
        pygame.time.delay(int(duration_secs * 1000))

    model = load_best_model()

    if model is None:
        raise Exception("No model is found.")

    FPS = 4

    pygame.init()

    gestures = sorted(['rock', 'paper', 'scissors'])  # , 'paper', 'scissors'
    lb = preprocessing.LabelBinarizer()
    lb.fit(gestures)

    last_pred_str = ""

    t2 = time.time_ns()
    inference_time = 0

    desired_processing_time = 1/FPS

    while True:
        last_buf = None

        while not inference_buffer.empty():
            last_buf = inference_buffer.get()

        if last_buf is not None:
            buf = np.array(last_buf)
            buf = transform_data_for_inference(buf)

            t0 = time.time()
            predictions = model.predict([buf])
            t1 = time.time()
            inference_time = t1-t0
            # print("Inference time: ", inference_time) # ~35 FPS

            processing_time = t1-t2

            time_diff = processing_time - desired_processing_time
            actual_fps = 1 / processing_time
            fps_str = f"Actual FPS: {actual_fps:.2f}"
            if time_diff > 0:
                fps_str += f" (Lag: {FPS - actual_fps:2f})"

            # print(fps_str)

            print(predictions)
            pred_dec = lb.inverse_transform(predictions)
            pred_str = pred_dec[0]
            if last_pred_str != pred_str or True:
                last_pred_str = pred_str
                print(pred_str, predictions[0][gestures.index(pred_str)])

            # z = list(zip(gestures, predictions[0]))
            # print(z)

        t2 = time.time()
        sleep(max(0.0, desired_processing_time - inference_time))

if __name__ == '__main__':
    emgQueue = multiprocessing.Queue()
    inference_buffer = multiprocessing.Queue()

    processRealtimeInference = Process(target=model_test_realtime, args=(emgQueue, inference_buffer), daemon=True)
    processRealtimeInference.start()

    processRealtimeLoop = Process(target=realtime_loop, args=(inference_buffer,), daemon=True)
    processRealtimeLoop.start()

    imuQueues = []
    emgQueues = []

    mainLoopThread = MainLoopThread(imuQueues, emgQueues, print_hz=False)

    emgQueues.append(emgQueue)
    isRunning = True

    processRealtimeInference.join()
    processRealtimeLoop.join()