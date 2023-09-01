## IMPORTS
import board
import time
from adafruit_bme280 import basic as adafruit_bme280

##CONFIGURATION VARIABLES
i2c = board.I2C()   # uses board.SCL and board.SDA  
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)

##VARIABLES
bme280.sea_level_pressure = 1013.25
temperature = bme280.temperature
humidity = bme280.relative_humidity
pressure = bme280.pressure
altitude = bme280.altitude

##METHODS
def getTemperature():
    return temperature

def getHumidity():
    return humidity

def getPressure():
    return pressure

def getAltitude():
    return altitude

def getAll():
    return temperature, humidity, pressure, altitude
