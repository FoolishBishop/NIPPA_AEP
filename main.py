import multiprocessing as mp

from ICM20948 import *
from BME280 import *

if __name__ == '__main__':
    #define sensors
    bme = BME280()
    icm = ICM20948()
    while True:
        #write to csv file in parallel
        p1 = mp.Process(target = bme.to_csv())
        p2 = mp.Process(target = icm.to_csv())

        p1.start()
        p2.start()

        #define other functionalities
