# Este es SOLO un programa para probar al acelerometro, esta no es la version final del programa.

from imu import MPU6050  # https://github.com/micropython-IMU/micropython-mpu9x50
import time
from machine import Pin, I2C

# Se necesita vector3d.py e imu.py del git de arriba
#   Pines:
#       VCC: No importa, (2 o 4)
#       GND: No importa (9, 25, 39, 6, 14, 20, 30, 34)
#       SCL: 3
#       SDA: 5

archive = open("data.txt", "w")

i2c = I2C(0, sda=Pin(2), scl=Pin(3), freq=400000)
imu = MPU6050(i2c)

print("***IMPORTANTE: Los valores de aceleracion, giroscopio y temperatura fueron redondeados***\n***Primera lectura "
      "inicia en 5 segundos***")
time.sleep(5)

while True:
    # reading values
    acceleration = imu.accel
    gyroscope = imu.gyro
    acele = [acceleration.x, acceleration.y, acceleration.z]
    print(f"Aceleracion\n X: {round(acele[0])}\n Y: {round(acele[1])}\n Z: {round(acele[2])}")
    giros = [gyroscope.x, gyroscope.y, gyroscope.z]
    print(f"Gyroscope\n X: {round(giros[0])}\n Y: {round(giros[1])}\n Z: {round(giros[2])}")
    temp = imu.temperature
    print("Temperature: ", round(temp, 2), " C")

    archive.write(f"{acele}\n{giros}\n{temp}")
    time.sleep(3)
    print("Reaing values in 2 seconds...")
    time.sleep(2)
