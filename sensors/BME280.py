from adafruit_bme280 import basic as adafruit_bme280
import multiprocessing as mp
import board
from _base import Sensor


class BME280(Sensor):
    def __init__(self):
        super().__init__()
        ##CONFIGURATION VARIABLES
        self.bme = adafruit_bme280.Adafruit_BME280_I2C(self.i2c)
        self.bme.sea_level_pressure = (
            1013.25  # change this to match the location's pressure (hPa) at sea level
        )
        self.columns = ["Temperature", "Humidity", "Pressure", "Altitude"]

    def get_data(self):
        temperature = self.bme.temperature
        humidity = self.bme.relative_humidity
        pressure = self.bme.pressure
        altitude = self.bme.altitude
        data = (temperature, humidity, pressure, altitude)
        self.queue.put(data)

    def display_data(self):
        print("\nTemperature: %0.1f C" % self.bme.temperature)
        print("Humidity: %0.1f %%" % self.bme.relative_humidity)
        print("Pressure: %0.1f hPa" % self.bme.pressure)
        print("Altitude = %0.2f meters" % self.bme.altitude)
