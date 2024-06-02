from datetime import datetime
import multiprocessing as mp
import time
import os
from sensors._base import Sensor
from sensors.ICM20948 import *
from sensors.BME280 import *
from sensors.PICAMERA import *
from lora.lora import *
from typing import Sequence, List, Any, Tuple
from itertools import chain
import argparse


class Main:
    def __init__(self, root_path: str) -> None:
        # for cronometer porpuses
        self.time = time.time()
        self.date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.root_path = root_path
        self.csv_path = f"{root_path}/{self.date}/csv/"
        os.makedirs(self.csv_path, exist_ok=True)
        os.makedirs(self.video_path, exist_ok=True)

        self.sensors: Sequence[Sensor] = [ICM20948(), BME280(), PICAMERA(root_path)]

        # Creates the columns for our data
        with open(self.csv_path, "a") as file:
            file.write(
                ",".join(
                    list(
                        chain.from_iterable([sensor.columns for sensor in self.sensors])
                    )
                )
                + "\n"
            )

        self.receptor = LORA()

    def get_data(self) -> Tuple[float | Any]:
        process: List[mp.Process] = []
        tabular: List[float | Any] = [time.time() - self.time]

        # Define process
        for sensor in self.sensors:
            process.append(mp.Process(target=sensor.get_data))

        # start process
        for proc in process:
            proc.start()

        # join
        for proc in process:
            proc.join()

        for sensor in self.sensors:
            data = sensor.queue.get()
            if not isinstance(data, np.array):
                tabular.append(data)
            else:
                sensor.save_data(data)

        return tuple(chain.from_iterable(tabular))

    def write(self, data: str) -> None:
        with open(self.csv_root, "a") as file:
            file.write(data + "\n")

    def __call__(self) -> None:
        while True:
            tabular_args = self.get_data()
            data = ",".join(tabular_args)
            self.receptor.send_data(data)
            self.write(data)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Cloud Chamber program")
    parser.add_argument("root", help="Storage path for data.")
    args = parser.parse_args()
    # Init program
    main = Main(args.root)
    main()
