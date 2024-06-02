# implement pi camera
import picamera
from picamera.array import PiRGBArray
from _base import Sensor
from matplotlib import pyplot as plt
import numpy as np
import os


class PICAMERA(Sensor):
    def __init__(self, root_path: str) -> None:
        super().__init__()
        self.columns = None
        self.camera = picamera.PiCamera()
        self.camera.resolution = (512, 512)
        self.camera.iso = 1200
        self.camera.shutter_speed = 1000
        self.camera.exposure_mode = "night"
        self.npy_path = lambda time: os.path.join(root_path, f"{time}.npy")

    def get_data(self):
        self.camera.capture()
        with picamera.array.PiRGBArray(self.camera) as data:
            self.camera.capture(data, "rgb")
            data = np.frombuffer(data, dtype=np.uint8, count=512 * 512).reshape(
                3, 512, 512
            )
            self.queue.put(data)

    def display_data(self) -> None:
        self.camera.capture()
        with picamera.array.PiRGBArray(self.camera) as data:
            self.camera.capture(data, "rgb")
            data = np.frombuffer(data, dtype=np.uint8, count=512 * 512).reshape(
                3, 512, 512
            )
            plt.imshow(data)

    def save_data(self, data: np.array, time: float) -> None:
        np.save(self.npy_path(time), data)
