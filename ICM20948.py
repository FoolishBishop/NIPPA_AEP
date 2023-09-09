# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_icm20x
from csv import writer
class ICM20948():
    def __init__(self):
        self.i2c = board.I2C()
        self.icm = adafruit_icm20x.ICM20948(self.i2c)
    def get_data(self):
            ax,ay,az = self.icm.acceleration
            gx,gy,gz = self.icm.gyro
            bx,by,bz = self.icm.magnetic
            return ax,ay,az,gx,gy,gz,bx,by,bz
    def display_data(self):
        print("Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (self.icm.acceleration))
        print("Gyro X:%.2f, Y: %.2f, Z: %.2f rads/s" % (self.icm.gyro))
        print("Magnetometer X:%.2f, Y: %.2f, Z: %.2f uT" % (self.icm.magnetic))
        print("")
    def to_csv(self):
        ax,ay,az,gx,gy,gz,bx,by,bz = self.get_data()
        with open("data/thpa.csv","a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([ax,ay,az,gx,gy,gz,bx,by,bz])
            f_object.close()