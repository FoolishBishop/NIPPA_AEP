from digitalio import DigitalInOut, Direction, Pull
import adafruit_ssd1306
import adafruit_rfm9x
import busio
import board
import time
import os
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Receiver:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        CS = DigitalInOut(board.CE1)
        RESET = DigitalInOut(board.D25)
        spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        self.receiver = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)
        self.receiver.tx_power = 23
        os.makedirs('data', exist_ok=True)

        self.figures = {}  
        self.axes = {}     

        self.data_keys = ['Temperature', 'Humidity', 'Pressure', 'Altitude', 
                          'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z',
                          'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z',
                          'Magnetic_X', 'Magnetic_Y', 'Magnetic_Z',]
        # Initialize plots for each data type
        for keys in self.data_keys:
            self.init_plot(keys, f'{keys}', f'{keys} over Time')

    def init_plot(self, key, ylabel, title):
        self.figures[key], self.axes[key] = plt.subplots()
        self.axes[key].set_ylabel(ylabel)
        self.axes[key].set_title(title)
        self.axes[key].tick_params(rotation=45, ha='right')
        self.axes[key].legend()

    def animate(self, i):
        packet = self.receiver.receive()
        if packet is not None:
            packet_text = str(packet, 'utf-8')
            received_data = [float(value) for value in packet_text.split(',')][1:]
            timestamp = dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]

            for idx, value in enumerate(received_data):
                self.plot_data(value, timestamp, received_data[idx])

            with open('data/data.csv', 'a') as file:
                file.write(packet_text + '\n')

    def plot_data(self, key, timestamp, value):
        if key in self.axes:
            if key not in self.axes[key].lines:
                self.axes[key].plot([], [], label=key)
            self.axes[key].lines[0].set_xdata(self.axes[key].lines[0].get_xdata() + [timestamp])
            self.axes[key].lines[0].set_ydata(self.axes[key].lines[0].get_ydata() + [value])

    def receive(self):
        ani = animation.FuncAnimation(list(self.figures.values())[0], self.animate, interval=500)
        plt.show()


if __name__ == '__main__':
    receiver = Receiver()
    receiver.receive()
