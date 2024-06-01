from digitalio import DigitalInOut, Direction, Pull
import adafruit_ssd1306
import adafruit_rfm9x
import busio
import board
import time


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

    def send_data(self, data: str):
        lista = data.split(",")
        for idx, k in enumerate(lista):
            lista[idx] = round(float(k), ndigits=3)

        button_a_data = bytes(str(lista), "utf-8")

        self.sender.send(button_a_data)
