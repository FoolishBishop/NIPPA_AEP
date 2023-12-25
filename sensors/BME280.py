from adafruit_bme280 import basic as adafruit_bme280
import multiprocessing as mp
import board

class BME280():
    def __init__(self):
        ##CONFIGURATION VARIABLES
        self.i2c = board.I2C()   # uses board.SCL and board.SDA
        self.bme = adafruit_bme280.Adafruit_BME280_I2C(self.i2c)
        self.bme.sea_level_pressure = 1013.25 # change this to match the location's pressure (hPa) at sea level
        self.bme_queue = mp.Queue()
    def get_data_bme(self):
        temperature = self.bme.temperature
        humidity = self.bme.relative_humidity
        pressure = self.bme.pressure
        altitude = self.bme.altitude
        data = (temperature, humidity, pressure, altitude)
        self.bme_queue.put(data)
    def display_data_bme(self):
        print("\nTemperature: %0.1f C" % self.bme.temperature)
        print("Humidity: %0.1f %%" % self.bme.relative_humidity)
        print("Pressure: %0.1f hPa" % self.bme.pressure)
        print("Altitude = %0.2f meters" % self.bme.altitude)
