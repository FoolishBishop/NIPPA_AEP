import multiprocessing as mp
import shutil
import time
import os

from sensors.ICM20948 import *
from sensors.BME280 import *
from sensors.PICAMERA import *
from lora.lora import *


class Sensors(ICM20948, BME280, LORA):
    def __init__(self):
        #Call the constructors of all sensors
        ICM20948.__init__(self)
        BME280.__init__(self)
        LORA.__init__(self)
        PICAMERA.__init__(self)
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
        temperature, humidity, pressure, altitude = self.bme_queue.get()
        
        ax,ay,az,gx,gy,gz,bx,by,bz = self.icm_queue.get()
        
        data = f'{time.time()-self.time},{temperature},{humidity},{pressure},{altitude},{ax},{ay},{az},{gx},{gy},{gz},{bx},{by},{bz}'

        self.send_data(data)

        #into the csv file
        with open("data/data.csv","a") as file:
            file.write(data+'\n')


if __name__ == '__main__':
    timer = time.time()
    #Creates data directory
    try:
        os.makedirs('data', exist_ok=False)
    except FileExistsError:
        #shoots down the directory if already exists and recreates it
        shutil.rmtree('data')
        os.makedirs('data')
    #Creates sensors object
    sensors = Sensors()
    
    p1 = mp.Process(target = sensors.get_data_camera)

    p1.start()
    
    p1.join()
    
    while True:
        p2 = mp.Process(target = sensors.to_csv)

        p2.start()
        
        p2.join()  

        if time.time() - timer > 3*3600:
            

            
