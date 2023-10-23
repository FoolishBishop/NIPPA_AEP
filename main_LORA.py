#setup LORALORALORALORALORAL -------------------------------------------------

# SPDX-FileCopyrightText: 2018 Brent Rubell for Adafruit Industries
#
# SPDX-License-Identifier: MIT

"""
Example for using the RFM9x Radio with Raspberry Pi.

Learn Guide: https://learn.adafruit.com/lora-and-lorawan-for-raspberry-pi
Author: Brent Rubell for Adafruit Industries
"""
# Import Python System Libraries
import time
# Import Blinka Libraries
import busio
from digitalio import DigitalInOut, Direction, Pull
import board
# Import the SSD1306 module.
import adafruit_ssd1306
# Import RFM9x
import adafruit_rfm9x

# Button A
btnA = DigitalInOut(board.D5)
btnA.direction = Direction.INPUT
btnA.pull = Pull.UP

# Button B
btnB = DigitalInOut(board.D6)
btnB.direction = Direction.INPUT
btnB.pull = Pull.UP

# Button C
btnC = DigitalInOut(board.D12)
btnC.direction = Direction.INPUT
btnC.pull = Pull.UP

# Create the I2C interface.
i2c = busio.I2C(board.SCL, board.SDA)

# 128x32 OLED Display
reset_pin = DigitalInOut(board.D4)
display = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c, reset=reset_pin)
# Clear the display.
display.fill(0)
display.show()
width = display.width
height = display.height

# Configure LoRa Radio
CS = DigitalInOut(board.CE1)
RESET = DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)
rfm9x.tx_power = 23
prev_packet = None

"""
while True:
    packet = None
   

    

    if not btnA.value:
        # Send text
   
        button_a_data = bytes("HOY GANA\r\n","utf-8")
        rfm9x.send(button_a_data)
  

    display.show()
    time.sleep(0.1)
"""



#MAIN ----------------------------------------------------------------- 
import multiprocessing as mp
import shutil
import time
import os

from sensors.ICM20948 import *
from sensors.BME280 import *

class Sensors(ICM20948, BME280):
    def __init__(self):
        #Call the constructors of all sensors
        ICM20948.__init__(self)
        BME280.__init__(self)
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
        
        #into the csv file
        
        with open("data/data.csv","a") as file:
            file.write(f'{time.time()-self.time},{temperature},{humidity},{pressure},{altitude},{ax},{ay},{az},{gx},{gy},{gz},{bx},{by},{bz}\n')
            

       #send with lora
        string = f'{time.time()-self.time},{temperature},{humidity},{pressure},{altitude},{ax},{ay},{az},{gx},{gy},{gz},{bx},{by},{bz}'
        lista = string.split(',')
        for idx, k in enumerate(lista):
            lista[idx] = round(float(k), ndigits = 3)
        
        button_a_data = bytes(str(lista),"utf-8")
        rfm9x.send(button_a_data)
        
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
        

