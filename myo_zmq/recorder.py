import ctypes
import multiprocessing
from pathlib import Path

from pyee import BaseEventEmitter
import numpy as np
import pandas as pd
import os

from typing import List


class Recording:
    data = []

    def __init__(self, name, full_file_path, data=None):
        if data is None:
            data = []
        self.name = name
        self.full_file_path = full_file_path
        self.data = data

    def fill_from_file(self, **params):
        self.data = pd.read_csv(self.full_file_path, **params)

    def add(self, data_to_add):
        self.data.append(data_to_add)

    def get_data(self):
        return np.array(self.data)

    def get_data_count(self):
        return len(self.data)

    def get_name(self):
        return self.name

    def get_file_path(self):
        return self.full_file_path


class Recorder(BaseEventEmitter):
    isRecording: bool = False
    lastRecording: Recording = None
    
    def __init__(self, save_directory: str, desired_recording_length: int = -1, file_format: str = "csv"):
        super(Recorder, self).__init__()
        self.saveDirectory = save_directory
        self.fileFormat = file_format
        self.desiredRecordingLength = desired_recording_length

    @staticmethod
    def get_file_path(directory, file_name, file_format) -> str:
        is_existing = os.path.exists(directory)
        if not is_existing:
            os.makedirs(directory)
            print(f"A new directory is created at: {directory}")

        i = 0
        get_full_path = lambda: os.path.join(directory, f"{file_name}_{i}.{file_format}")
        while os.path.exists(get_full_path()):
            i += 1

        return get_full_path()

    @staticmethod
    def get_recordings(directory, file_name, file_format) -> List[Recording]:
        recordings = []
        i = 0
        get_full_path = lambda: os.path.join(directory, f"{file_name}_{i}.{file_format}")
        while os.path.exists(get_full_path()):
            path = get_full_path()
            new_recording = Recording(path, data=pd.read_csv(path, header=None).as_matrix())
            recordings.append(new_recording)
            i += 1

        return recordings

    def begin_recording(self, file_name: str, force=False):
        if self.isRecording:
            if not force:
                print("There is an ongoing recording. Use 'force' parameter to stop recording and start anew.")
                return
            self.end_recording(reset=True)

        try:
            full_file_path = self.get_file_path(self.saveDirectory, file_name, self.fileFormat)
            self.lastRecording = Recording(file_name, full_file_path)
        except Exception as e:
            print("Error", e)
            self.reset()
            raise

        self.isRecording = True
        self.emit("begin_recording")

    def add(self, data_to_add):
        if not self.isRecording:
            print("There is no recording in process to add data to.")
            return

        if self.desiredRecordingLength > 0:
            if self.lastRecording.get_data_count() + 1 >= self.desiredRecordingLength:
                print("Desired recording length reached")
                self.save_recording()
                return

        self.lastRecording.add(data_to_add)

    def end_recording(self, reset=False, silent=False):
        if not self.isRecording:
            print("There is no recording in process to stop.")
            return

        if not silent:
            print("Ended recording: ", self.lastRecording.get_file_path())
            self.emit("end_recording")

        if reset:
            self.reset()


    def save_recording(self):
        if self.isRecording:
            self.end_recording(reset=False, silent=True)

        if self.lastRecording is not None:
            data = self.lastRecording.get_data()
            df = pd.DataFrame(data)
            file_path = self.lastRecording.get_file_path()
            df.to_csv(Path(file_path), header=False, index=False)
            print(f"Saved recording to file '{self.lastRecording.get_file_path()}', with duration of {len(data)} frames")
            self.emit("save_recording")
        self.reset()

    def reset(self):
        self.isRecording = False
        self.lastRecording = None
