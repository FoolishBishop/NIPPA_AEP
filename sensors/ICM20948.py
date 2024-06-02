import adafruit_icm20x
import board
import multiprocessing as mp
from _base import Sensor


class ICM20948(Sensor):
    def __init__(self):
        self.icm = adafruit_icm20x.ICM20948(self.i2c)
        self.columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Bx", "By", "Bz"]

    def get_data(self):
        ax, ay, az = self.icm.acceleration
        gx, gy, gz = self.icm.gyro
        bx, by, bz = self.icm.magnetic
        data = (ax, ay, az, gx, gy, gz, bx, by, bz)
        self.queue.put(data)

    def display_data(self):
        print("Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (self.icm.acceleration))
        print("Gyro X:%.2f, Y: %.2f, Z: %.2f rads/s" % (self.icm.gyro))
        print("Magnetometer X:%.2f, Y: %.2f, Z: %.2f uT" % (self.icm.magnetic))
        print("")
