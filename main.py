import multiprocessing as mp
import shutil
import time
import os

from sensors.ICM20948 import *
from sensors.BME280 import *

class Sensors(ICM20948, BME280):
    def __init__(self):
        #Call the constructors of all sensors
        super(BME280, self).__init__()
        super(ICM20948, self).__init__()

        """
        If you want to add sensors:
        super(<sensor_class>, self).__init__(paramters if exists)

        add a third process to to_csv and change the output into the csv file

        change the column names on init constructor in order to match the new data

        you are ready to go
        """
        #for cronometer porpuses
        self.time = time.time()
        #Creates the columns for our data
        with open("data/data.csv","a") as file:
             file.write('time,Temperature,Humidity,Pressure,Altitude,Ax,Ay,Az,Gx,Gy,Gz,Bx,By,Bz\n')
    def to_csv(self):
        #Creates subprocesses in parallel
        p1 = mp.Process(target = self.get_data_bme)
        p2 = mp.Process(target = self.get_data_icm)

        p1.start()
        p2.start()
        
        p1.join()
        p2.join()        
        
        #get data
        temperature, humidity, pressure, altitude = p1.get()
        
        ax,ay,az,gx,gy,gz,bx,by,bz = p2.get()
        
        #into the csv file
        with open("data/data.csv","a") as file:
            file.write(f'{time.time()-self.time},{temperature},{humidity},{pressure},{altitude},{ax},{ay},{az},{gx},{gy},{gz},{bx},{by},{bz}\n')

if __name__ == '__main__':
    #Creates data directory
    try:
        os.makedirs('data', exist_ok=False)
    except FileExistsError:
        #shoots down the directory if already exists and recreates it
        shutil.rmtree('data')
        os.makedirs('data')
    #Creates sensors object
    sensors = Sensors()
    while True:
        #returns the data into .csv format
        sensors.to_csv()
