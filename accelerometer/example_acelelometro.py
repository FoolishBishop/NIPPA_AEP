from imu import MPU6050  # https://github.com/micropython-IMU/micropython-mpu9x50
import time
from machine import Pin, I2C

# Se necesita vector3d.py y imu.py de la pagina de arriba
#   Pines:
#       VCC: No importa, (2 o 4)
#       GND: No importa (9, 25, 39, 6, 14, 20, 30, 34)
#       SCL: 3
#       SDA: 5

i2c = I2C(0, sda=Pin(2), scl=Pin(3), freq=400000)
imu = MPU6050(i2c)

# Temperature display
# Los acelerometros pueden medir temperatura (???)
print("Temperature: ", round(imu.temperature, 2), " C")

while True:
    # reading values
    acceleration = imu.accel
    gyroscope = imu.gyro

    # creo que se podrian reemplazar los if por switch(?)
    if gyroscope.x > 45:
        print("Rotation left")

    if gyroscope.x < -45:
        print("Rotation right")

    if gyroscope.y > 45:
        print("Rotation forward")

    if gyroscope.z > 45:
        print("Twist left")

    if gyroscope.z < -45:
        print("Twist right")

    time.sleep(0.2)
