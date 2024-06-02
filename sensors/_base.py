import board
import multiprocessing as mp
from typing import Tuple, Any


class Sensor:
    def __init__(self) -> None:
        self.i2c = board.I2C()
        self.queue = mp.Queue()

    def get_data(self) -> Tuple[float, ...] | Any:
        raise NotImplementedError("Must define get_data method inside the sensor")

    def display_data(self) -> None:
        raise NotImplementedError("Must define display_data method inside the sensor")
