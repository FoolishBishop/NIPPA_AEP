import board
import time
from adafruit_bme280 import basic as adafruit_bme280
from csv import writer
class BME280():
    def __init__(self):
        ##CONFIGURATION VARIABLES
        i2c = board.I2C()   # uses board.SCL and board.SDA
        self.sensor = adafruit_bme280.Adafruit_BME280_I2C(i2c)
        self.sensor.sea_level_pressure = 1013.25 # change this to match the location's pressure (hPa) at sea level
    def get_data(self):
        temperature = self.sensor.temperature
        humidity = self.sensor.relative_humidity
        pressure = self.sensor.pressure
        altitude = self.sensor.altitude
        return temperature, humidity, pressure, altitude
    def display_data(self):
        print("\nTemperature: %0.1f C" % self.sensor.temperature)
        print("Humidity: %0.1f %%" % self.sensor.relative_humidity)
        print("Pressure: %0.1f hPa" % self.sensor.pressure)
        print("Altitude = %0.2f meters" % self.sensor.altitude)
    def write_to_csv(self):
        with open("data/thpa.csv","a") as f_object:
            writer_object = writer(f_object)
            temperature, humidity, pressure, altitude = self.get_data()
            writer_object.writerow([temperature, humidity, pressure, altitude])
            f_object.close()
