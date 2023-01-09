import asyncio
import itertools
import time
import zmq.asyncio
from dynamic_plot import Plot
import multiprocessing
from multiprocessing import Process
import numpy as np
import json
from myo_zmq.common import ZMQ_Topic

import matplotlib as mpl

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

def remap(old_val, old_min, old_max, new_min, new_max): return (new_max - new_min)*(old_val - old_min) / (old_max - old_min) + new_min

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

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

    def process_data(epoch_time, emg_data):
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

def runGraphIMU(imu_queue):
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


async def Loop(imu_queue, emg_queue):
    imu_t0 = time.time()
    emg_t0 = time.time()
    imu_count = 0
    emg_count = 0
    while True:
        json_message = json.loads(socket.recv_json())
        topic = json_message["topic"]
        data = json_message["data"]
        epoch_time = json_message["time"]

        data['time'] = epoch_time

        if topic == ZMQ_Topic.IMU:
            imu_queue.put(data)
            # print(packet)
            time_now = time.time()
            imu_count = imu_count + 1
            if time_now-imu_t0 > 1:
                imu_t0 = time.time()
                print("IMU Hz: ", imu_count)
                imu_count = 0

        elif topic == ZMQ_Topic.EMG:
            emg_queue.put(data)

            time_now = time.time()
            emg_count = emg_count + 1
            if time_now - emg_t0 > 1:
                emg_t0 = time.time()
                print("EMG Hz: ", emg_count)
                emg_count = 0


if __name__ == '__main__':
    host = 'raspberrypi.local'
    port = 8358
    url = 'tcp://' + host + ':' + str(port)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(url)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.RCVHWM, 1000)

    ctx = zmq.asyncio.Context()

    imu_queue = multiprocessing.Queue()
    emg_queue = multiprocessing.Queue()

    p_IMU = Process(target=runGraphIMU, args=(imu_queue, ))
    p_IMU.start()

    p_EMG = Process(target=runGraphEMG, args=(emg_queue, ))
    p_EMG.start()

    asyncio.run(Loop(imu_queue, emg_queue))
    p_IMU.join()
    p_EMG.join()