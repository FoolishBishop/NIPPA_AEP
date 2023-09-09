from icm20948 import ICM20948
from csv import writer
import sqlite3


class ICM_20948(ICM20948):
    def to_csv(self):
        bx, by, bz = self.read_magnetometer_data()       
        ax, ay, az, gx, gy, gz = self.read_accelerometer_gyro_data()
        with open('data/agm.csv') as file:
            writer_object = writer(file)
            writer_object.writerow([bx, by, bz, ax, ay, az, gx, gy, gz])
            file.close()


