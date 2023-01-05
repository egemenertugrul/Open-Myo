import math
import open_myo as myo
import time
import json
import zmq
from myo_zmq.common import ZMQ_Topic

def send(topic, data):
    if socket is None:
        return
    socket.send_json(json.dumps({"topic": topic, "data": data, "time": time.time()}))


def process_emg(emg):
    send(ZMQ_Topic.EMG, {"emg": emg})
    # print(emg)

def process_imu(quat, acc, gyro):
    send(ZMQ_Topic.IMU, {"acc": acc, "gyro": gyro, "quat": quat})
    # print(acc)
    # print(gyro)
    # print(quat)

def process_sync(arm, x_direction):
    print(arm, x_direction)

def process_classifier(pose):
    # send(topic["imu"], pose)
    print(pose)

def process_battery(batt):
    print("Battery level: %d" % batt)

# def led_emg(emg):
#     if(emg[0] > 80):
#         myo_device.services.set_leds([255, 0, 0], [128, 128, 255])
#     else:
#         myo_device.services.set_leds([128, 128, 255], [128, 128, 255])


myo_mac_addr = "C5:0C:80:21:19:3C"

context = zmq.Context()
port = 8358
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)

emgMode = myo.EmgMode.FILT
imuMode = myo.ImuMode.DATA
classifierMode = myo.ClassifierMode.ON


if __name__ == '__main__':
    print("MAC address: %s" % myo_mac_addr)
    myo_device = myo.Device(mac=myo_mac_addr)
    myo_device.services.sleep_mode(1)  # never sleep
    myo_device.services.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs)

    fw = myo_device.services.firmware()
    print("Firmware version: %d.%d.%d.%d" % (fw[0], fw[1], fw[2], fw[3]))
    batt = myo_device.services.battery()
    print("Battery level: %d" % batt)
    for i in range(math.ceil(batt/20)):
        myo_device.services.vibrate(1)  # short vibration

    if emgMode == myo.EmgMode.FILT:
        myo_device.services.emg_filt_notifications()
    elif emgMode == myo.EmgMode.RAW:
        myo_device.services.emg_raw_notifications()

    myo_device.services.imu_notifications()
    myo_device.services.classifier_notifications()
    # myo_device.services.battery_notifications()
    myo_device.services.set_mode(emgMode, imuMode, classifierMode)
    myo_device.add_emg_event_handler(process_emg)
    # myo_device.add_emg_event_handler(led_emg)
    myo_device.add_imu_event_handler(process_imu)
    # myo_device.add_sync_event_handler(process_sync)
    # myo_device.add_classifier_event_hanlder(process_classifier)

    myo_device.add_battery_event_handler(process_battery)

    while True:
        if myo_device.services.waitForNotifications(1):
            # time.sleep(1)
            continue
        print("Waiting...")

