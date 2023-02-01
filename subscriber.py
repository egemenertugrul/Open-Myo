from __future__ import annotations

import asyncio
import itertools
import os
import sys
import time
from typing import List

import zmq.asyncio
from dynamic_plot import Plot
import multiprocessing
from multiprocessing import Process
import numpy as np
import json
from myo_zmq.common import ZMQ_Topic
from myo_zmq.recorder import Recording, Recorder
import threading

import matplotlib as mpl


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


def remap(old_val, old_min, old_max, new_min, new_max): return (new_max - new_min) * (old_val - old_min) / (
        old_max - old_min) + new_min


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def runGraphEMG(emg_queue):
    d_plot_2 = Plot(rowcol=(2, 4), max_display_capacity=500, ylim=(-150, 150))

    listen_significant_changes = True

    d_plot_2.set_title((1, 1), "EMG 1")
    d_plot_2.set_title((1, 2), "EMG 2")
    d_plot_2.set_title((1, 3), "EMG 3")
    d_plot_2.set_title((1, 4), "EMG 4")
    d_plot_2.set_title((2, 1), "EMG 5")
    d_plot_2.set_title((2, 2), "EMG 6")
    d_plot_2.set_title((2, 3), "EMG 7")
    d_plot_2.set_title((2, 4), "EMG 8")

    if listen_significant_changes:
        mean_list_size = 10
        recorded_data_list = []
        last_means = []
        threshold = 5

    red = "#ff0000"
    green = "#00ff00"
    blue = "#0000ff"

    def process_data(epoch_time, emg_data: multiprocessing.Queue):
        nonlocal last_means

        if listen_significant_changes:
            recorded_data_list.append(emg_data)
            if len(recorded_data_list) > mean_list_size:
                recorded_data_list.pop(0)

            cur_means = np.mean(recorded_data_list, axis=0)
            significant_change_list = None
            diff = None
            if len(last_means):
                diff = cur_means - last_means
                significant_change_list = np.abs(diff) > threshold
            last_means = cur_means

            if diff is not None and significant_change_list is not None:
                for (i, j) in itertools.product(range(d_plot_2.row_count), range(d_plot_2.column_count)):
                    idx = i * d_plot_2.row_count + j
                    is_significant_change = significant_change_list[idx]
                    difference = diff[idx]
                    is_diff_positive = difference >= 0

                    # if is_significant_change:
                    color = blue

                    scale = clamp(abs(difference) / threshold, 0, 1)

                    if is_diff_positive:
                        # color = "#00ff00"
                        color = colorFader(blue, red, scale)
                    # else:
                    #     color = "#ff0000"
                    #     # color = colorFader(red, blue, scale)
                    d_plot_2.change_color((i + 1, j + 1), color)
                    # else:
                    #     d_plot_2.change_color((i+1, j+1), None)

        d_plot_2.add_to((1, 1), (epoch_time, emg_data[0]))
        d_plot_2.add_to((1, 2), (epoch_time, emg_data[1]))
        d_plot_2.add_to((1, 3), (epoch_time, emg_data[2]))
        d_plot_2.add_to((1, 4), (epoch_time, emg_data[3]))
        d_plot_2.add_to((2, 1), (epoch_time, emg_data[4]))
        d_plot_2.add_to((2, 2), (epoch_time, emg_data[5]))
        d_plot_2.add_to((2, 3), (epoch_time, emg_data[6]))
        d_plot_2.add_to((2, 4), (epoch_time, emg_data[7]))

    while True:
        while not emg_queue.empty():
            data = emg_queue.get()

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

        d_plot_2.flush()


def runGraphIMU(imu_queue: multiprocessing.Queue):
    take_mean = False

    d_plot = Plot(rowcol=(2, 3), max_display_capacity=500, style_args=
    [
        [
            ['r'], ['g'], ['b']
        ], [
        ['r-'], ['g-'], ['b-']
    ]
    ])

    d_plot.set_title((1, 1), "acc x")
    d_plot.set_title((1, 2), "acc y")
    d_plot.set_title((1, 3), "acc z")

    d_plot.set_title((2, 1), "gyro x")
    d_plot.set_title((2, 2), "gyro y")
    d_plot.set_title((2, 3), "gyro z")

    while True:
        acc_vals = []
        gyro_vals = []
        time_vals = []

        while not imu_queue.empty():
            data = imu_queue.get()
            epoch_time = float(data['time'])

            # print(data)
            acc = list(map(float, data['acc']))
            # acc = swapPositions(acc, pos1=0, pos2=1)

            gyro = list(map(float, data['gyro']))
            # print(time.time()-epoch_time)

            if take_mean:
                acc_vals.append(acc)
                gyro_vals.append(gyro)
                time_vals.append(epoch_time)
            else:
                d_plot.add_to((1, 1), (epoch_time, acc[0]))
                d_plot.add_to((1, 2), (epoch_time, acc[1]))
                d_plot.add_to((1, 3), (epoch_time, acc[2]))

                d_plot.add_to((2, 1), (epoch_time, gyro[0]))
                d_plot.add_to((2, 2), (epoch_time, gyro[1]))
                d_plot.add_to((2, 3), (epoch_time, gyro[2]))

        if take_mean and len(acc_vals) > 0:
            acc_vals = np.mean(acc_vals, axis=0)
            gyro_vals = np.mean(gyro_vals, axis=0)
            time_vals = time_vals[-1]

            d_plot.add_to((1, 1), (time_vals, acc_vals[0]))
            d_plot.add_to((1, 2), (time_vals, acc_vals[1]))
            d_plot.add_to((1, 3), (time_vals, acc_vals[2]))

            d_plot.add_to((2, 1), (time_vals, gyro_vals[0]))
            d_plot.add_to((2, 2), (time_vals, gyro_vals[1]))
            d_plot.add_to((2, 3), (time_vals, gyro_vals[2]))

        d_plot.flush()


# async def Loop(imu_queues: List[multiprocessing.Queue], emg_queues: List[multiprocessing.Queue]):
#     imu_t0 = time.time()
#     emg_t0 = time.time()
#     imu_count = 0
#     emg_count = 0
#     while True:
#         json_message = json.loads(socket.recv_json())
#         topic = json_message["topic"]
#         data = json_message["data"]
#         epoch_time = json_message["time"]
#
#         data['time'] = epoch_time
#
#         if topic == ZMQ_Topic.IMU:
#             for imu_queue in imu_queues:
#                 imu_queue.put(data)
#
#             # print(packet)
#             time_now = time.time()
#             imu_count = imu_count + 1
#             if time_now - imu_t0 > 1:
#                 imu_t0 = time.time()
#                 # print("IMU Hz: ", imu_count)
#                 imu_count = 0
#
#         elif topic == ZMQ_Topic.EMG:
#             for emg_queue in emg_queues:
#                 emg_queue.put(data)
#
#             time_now = time.time()
#             emg_count = emg_count + 1
#             if time_now - emg_t0 > 1:
#                 emg_t0 = time.time()
#                 # print("EMG Hz: ", emg_count)
#                 emg_count = 0


class MainLoopThread(threading.Thread):
    def __init__(self, imu_queues: List[multiprocessing.Queue], emg_queues: List[multiprocessing.Queue], print_hz: bool = True):
        super(MainLoopThread, self).__init__(daemon=True)
        self.imu_queues = imu_queues
        self.emg_queues = emg_queues
        self.print_hz = print_hz
        self.start()

    def run(self):
        host = 'raspberrypi.local'
        port = 8358
        url = 'tcp://' + host + ':' + str(port)
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(url)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        socket.setsockopt(zmq.RCVHWM, 1000)
        ctx = zmq.asyncio.Context()

        imu_t0 = time.time()
        emg_t0 = time.time()
        if self.print_hz:
            imu_count = 0
            emg_count = 0
        while True:
            json_message = json.loads(socket.recv_json())
            topic = json_message["topic"]
            data = json_message["data"]
            epoch_time = json_message["time"]

            data['time'] = epoch_time

            if topic == ZMQ_Topic.IMU:
                for imu_queue in self.imu_queues:
                    imu_queue.put(data)

                # print(packet)
                if self.print_hz:
                    time_now = time.time()
                    imu_count = imu_count + 1
                    if time_now - imu_t0 > 1:
                        imu_t0 = time.time()
                        print("IMU Hz: ", imu_count)
                        imu_count = 0

            elif topic == ZMQ_Topic.EMG:
                for emg_queue in self.emg_queues:
                    emg_queue.put(data)

                if self.print_hz:
                    time_now = time.time()
                    emg_count = emg_count + 1
                    if time_now - emg_t0 > 1:
                        emg_t0 = time.time()
                        print("EMG Hz: ", emg_count)
                        emg_count = 0


class KeyboardThread(threading.Thread):

    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            time.sleep(1)
            self.input_cbk(input())


def runRecorder(emg_queue: multiprocessing.Queue, keyboardInputQueue: multiprocessing.Queue):
    # recorder = Recorder(save_directory=os.path.join(os.getcwd(), "recordings"), file_format='.csv')
    recorder: Recorder = None

    def process_data(epoch_time, emg_data):
        if recorder:
            if recorder.isRecording:
                recorder.add(emg_data)

    while True:
        if not keyboardInputQueue.empty():
            inp_str = keyboardInputQueue.get()
            if len(inp_str) > 0:
                inp_str = inp_str.split(" ")
                cmd = inp_str[0]

                if cmd == "r" or cmd == "recorder":
                    relative_recording_directory = "recordings"
                    recording_len = -1
                    file_format = "csv"

                    if len(inp_str) > 1:
                        relative_recording_directory = str(inp_str[1])

                        if len(inp_str) > 2:
                            recording_len = int(inp_str[2])

                            if len(inp_str) > 3:
                                file_format = str(inp_str[3])

                    save_directory = os.path.join(os.getcwd(), relative_recording_directory)
                    recorder = Recorder(save_directory=save_directory,
                                        desired_recording_length=recording_len,
                                        file_format=file_format)

                    print(
                        f"Created recorder with save_directory: {save_directory}, desired_recording_length: {recording_len}, file_format: {file_format}")
                    print(f"To begin the recording,\n (b)egin\n\t<gesture_name (default: 'test_gesture', str)>")

                    def on_begin_recording():
                        print(f"To end and save the recording,\n (s)ave")
                        print(f"To end the recording,\n (e)nd")

                    def on_end_recording():
                        print(f"To save the recording,\n (s)ave")
                        print(
                            f"To discard the current one and begin a new recording,\n (b)egin\n\t<gesture_name ("
                            f"default: 'test_gesture', str)>")

                    def on_save_recording():
                        print(f"To begin the recording,\n (b)egin\n\t<gesture_name (default: 'test_gesture', str)>")

                    recorder.on('begin_recording', on_begin_recording)
                    recorder.on('end_recording', on_end_recording)
                    recorder.on('save_recording', on_save_recording)

                if cmd == "b" or cmd == "begin":
                    gesture_name = "gesture_name"
                    if len(inp_str) > 1:
                        gesture_name = str(inp_str[1])
                    if recorder:
                        recorder.begin_recording(gesture_name)

                if cmd == "e" or cmd == "end":
                    if recorder:
                        recorder.end_recording()

                if cmd == "s" or cmd == "save":
                    if recorder:
                        recorder.save_recording()

                # print(f"{inp}")

        while not emg_queue.empty():
            data = emg_queue.get()

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


def true_false_prompt(text: str, expected_vals: dict | None = None) -> bool:
    if text is None:
        print("Prompt text must be provided.")
    if expected_vals is None:
        expected_vals = {'y': True, 'n': False}
    while True:
        prompt_res = str(input(text + '\n')).lower()
        if prompt_res in expected_vals.keys():
            return expected_vals[prompt_res]


if __name__ == '__main__':
    imuQueues = []
    emgQueues = []
    all_processes = []

    isPlotIMU = true_false_prompt("Would you like to PLOT IMU data? (y/n)")

    if isPlotIMU:
        imuQueue = multiprocessing.Queue()
        imuQueues.append(imuQueue)
        p_IMU = Process(target=runGraphIMU, args=(imuQueue,), daemon=True)
        all_processes.append(p_IMU)
        p_IMU.start()

    isPlotEMG = true_false_prompt("Would you like to PLOT EMG data? (y/n)")

    if isPlotEMG:
        emgQueue = multiprocessing.Queue()
        emgQueues.append(emgQueue)
        p_EMG = Process(target=runGraphEMG, args=(emgQueue,), daemon=True)
        all_processes.append(p_EMG)
        p_EMG.start()

    isRecordEMG = true_false_prompt("Would you like to RECORD EMG data? (y/n)")

    if isRecordEMG:
        emgQueueToRecord = multiprocessing.Queue()
        emgQueues.append(emgQueueToRecord)

        keyboardInputQueue = multiprocessing.Queue()
        p_Recorder = Process(target=runRecorder, args=(emgQueueToRecord, keyboardInputQueue), daemon=True)
        all_processes.append(p_Recorder)
        p_Recorder.start()


        def keyboard_callback(inp):
            inp = str(inp).lower()
            keyboardInputQueue.put(inp)


        kthread = KeyboardThread(keyboard_callback)
        print(
            "(r)ecord \n\t<relative_recording_directory('default: recordings', str)>\n\t<recording_len ('default: -1', "
            "int)>\n\t<file_format ('default: csv', str)>")

    mainLoopThread = MainLoopThread(imuQueues, emgQueues, print_hz=not isRecordEMG)
    # asyncio.run(Loop([imuQueue], [emgQueue, emgQueueToRecord]))

    for process in all_processes:
        process.join()
