from digitalio import DigitalInOut, Direction, Pull
import adafruit_ssd1306
import adafruit_rfm9x
import busio
import board
import time
import os
#by Leonardo Rivarola
class Receiver:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        CS = DigitalInOut(board.CE1)
        RESET = DigitalInOut(board.D25)
        spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        self.receiver = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)
        self.receiver.tx_power = 23
        prev_packet = None
        os.makedirs('data', exist_ok = True)
    def receive(self):
        while True:
            packet = None

            packet = self.receiver.receive()
            if packet is None:
                print("")
            else:
                packet_text = str(packet, 'utf-8')
                print('RX: ', packet_text)
                with open('data/data.csv', 'a') as file:
                    file.write(packet_text + '\n')

if __name__=='__main__':
    receiver = Receiver()
    receiver.receive()
