import asyncio
import time
import zmq.asyncio
from dynamic_plot import Plot
import multiprocessing
from multiprocessing import Process
import numpy as np
import json
from myo_zmq.common import ZMQ_Topic

def runGraphEMG(emg_queue):
    d_plot_2 = Plot(rowcol=(2, 4), max_display_capacity=100)

    d_plot_2.set_title((1, 1), "EMG 1")
    d_plot_2.set_title((1, 2), "EMG 2")
    d_plot_2.set_title((1, 3), "EMG 3")
    d_plot_2.set_title((1, 4), "EMG 4")
    d_plot_2.set_title((2, 1), "EMG 5")
    d_plot_2.set_title((2, 2), "EMG 6")
    d_plot_2.set_title((2, 3), "EMG 7")
    d_plot_2.set_title((2, 4), "EMG 8")

    while True:
        while not emg_queue.empty():
            data = emg_queue.get()
            emg_data = list(map(float, data['emg']))
            epoch_time = float(data['time'])

            d_plot_2.add_to((1, 1), (epoch_time, emg_data[0]))
            d_plot_2.add_to((1, 2), (epoch_time, emg_data[1]))
            d_plot_2.add_to((1, 3), (epoch_time, emg_data[2]))
            d_plot_2.add_to((1, 4), (epoch_time, emg_data[3]))
            d_plot_2.add_to((2, 1), (epoch_time, emg_data[4]))
            d_plot_2.add_to((2, 2), (epoch_time, emg_data[5]))
            d_plot_2.add_to((2, 3), (epoch_time, emg_data[6]))
            d_plot_2.add_to((2, 4), (epoch_time, emg_data[7]))

        d_plot_2.flush()


def runGraphIMU(imu_queue):
    take_mean = True

    d_plot = Plot(rowcol=(2, 3), max_display_capacity=100, style_args=
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
    t0 = time.time()
    count = 0
    while True:
        json_message = json.loads(socket.recv_json())
        topic = json_message["topic"]
        data = json_message["data"]
        epoch_time = json_message["time"]

        data['time'] = epoch_time

        if topic == ZMQ_Topic.IMU:
            imu_queue.put(data)
            # print(packet)
            t1 = time.time()
            count = count + 1
            if t1-t0 > 1:
                t0 = time.time()
                print(count)
                count = 0

        elif topic == ZMQ_Topic.EMG:
            emg_queue.put(data)


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