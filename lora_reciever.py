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

        self.figs = {}  
        self.axes = {}     

        self.data_keys = ['Temperature', 'Humidity', 'Pressure', 'Altitude', 
                          'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z',
                          'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z',
                          'Magnetic_X', 'Magnetic_Y', 'Magnetic_Z']
        # Initialize plots for each data type
        for idx, key in enumerate(self.data_keys):
            self.init_plot(idx, f'{key}', f'{key} over Time')

    def init_plot(self, idx, ylabel, title):
        self.figs, self.axes = plt.subplots(3,5, figsize= (20,8))
        self.axes[idx].set_ylabel(ylabel)
        self.axes[idx].set_title(title)
        self.axes[idx].tick_params(rotation=45, ha='right')
        self.axes[idx].legend()

    def animate(self, i):
        packet = self.receiver.receive()
        if packet is not None:
            packet_text = str(packet, 'utf-8')
            received_data = [float(value) for value in packet_text.split(',')][1:]
            timestamp = dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]

            for idx, value in enumerate(self.data_keys):
                self.plot_data(value, timestamp, received_data[idx])

            with open('data/data.csv', 'a') as file:
                file.write(packet_text + '\n')

    def plot_data(self, key, timestamp, value):
        if key in self.axes:
            if key not in self.axes[key].lines:
                self.axes[key].plot([], [], label=key)
            self.axes[key].lines[0].set_xdata(self.axes[key].lines[0].get_xdata() + [timestamp])
            self.axes[key].lines[0].set_ydata(self.axes[key].lines[0].get_ydata() + [value])

    def showAnim(self):
        ani = animation.FuncAnimation(list(self.figs.values())[0], self.animate, interval=500)
        plt.show()


if __name__ == '__main__':
    receiver = Receiver()
    receiver.showAnim()
