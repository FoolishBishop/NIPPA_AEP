import time
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

class LORA:
    def __init__(self):
        # Create the I2C interface.
        i2c = busio.I2C(board.SCL, board.SDA)

        # Configure LoRa Radio
        CS = DigitalInOut(board.CE1)
        RESET = DigitalInOut(board.D25)
        spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        self.sender = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)
        self.sender.tx_power = 23
    def send_data(self,data: str):
        lista = data.split(',')
        for idx, k in enumerate(lista):
            lista[idx] = round(float(k), ndigits = 3)

        button_a_data = bytes(str(lista),"utf-8")
        
        self.sender.send(button_a_data)