import board
import time
from adafruit_bme280 import basic as adafruit_bme280

##CONFIGURATION VARIABLES
i2c = board.I2C()   # uses board.SCL and board.SDA
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
data = open("temperature.csv","a")

##VARIABLES
bme280.sea_level_pressure = 1013.25 # change this to match the location's pressure (hPa) at sea level
temperature = bme280.temperature
humidity = bme280.relative_humidity
pressure = bme280.pressure
altitude = bme280.altitude

##TEST DE INICIO --Borrar Despues--
print("\nTemperature: %0.1f C" % bme280.temperature)
print("Humidity: %0.1f %%" % bme280.relative_humidity)
print("Pressure: %0.1f hPa" % bme280.pressure)
print("Altitude = %0.2f meters" % bme280.altitude)

with data as f_object:
    writer_object = writer(f_object)
    writer_object.writerow([temperature, humidity, pressure, altitude])
    f_object.close()
