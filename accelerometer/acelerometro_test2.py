# ESTO NO SERA UTILIZADO AL FINAL ESTO ES SOLO TESTEO
# hacer mas eficiente y preferiblemente pasar a otro lenguaje

import smbus
import math
from time import sleep

# Se necesita vector3d.py e imu.py del git de arriba
#   Pines:
#       VCC: No importa, (2 o 4)
#       GND: No importa (9, 25, 39, 6, 14, 20, 30, 34)
#       SCL: 3
#       SDA: 5

# Guardar data
archive = open("data.txt", "w")

# some MPU6050 Registers and their Address
power_mgmt_1 = 0x6B
address = 0x68

bus = smbus.SMBus(1)


# sacar datos mas crudos que el asado de los domingos de aca
def read_byte(adr):
    return bus.read_byte_data(address, adr)


def read_word(adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr + 1)
    val = (high << 8) + low
    return val


def read_word_2c(adr):
    val = read_word(adr)
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val


# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# yo no tener inteligencia espacial que alguien me explique
def dist(a, b):
    return math.sqrt((a * a) + (b * b))


def get_y_rotation(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)


def get_x_rotation(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)


# Now wake the 6050 up as it starts in sleep mode
bus.write_byte_data(address, power_mgmt_1, 0)

print("*** Programa empezara a funcar en 3 segundos ***")
sleep(3)

while True:
    # Read Accelerometer raw value
    acc_x = read_word_2c(0x3b)
    acc_y = read_word_2c(0x3d)
    acc_z = read_word_2c(0x3f)

    # Read Gyroscope raw value
    gyro_x = read_word_2c(0x43)
    gyro_y = read_word_2c(0x45)
    gyro_z = read_word_2c(0x47)

    # Full scale range +/- 250 degree/C as per sensitivity scale factor
    Ax = acc_x / 16384.0
    Ay = acc_y / 16384.0
    Az = acc_z / 16384.0

    Gx = gyro_x / 131.0
    Gy = gyro_y / 131.0
    Gz = gyro_z / 131.0

    # no se como funca el primero pero lo dejo ahi nomas
    print("Gx=%.2f" % Gx, u'\u00b0' + "/s", "\tGy=%.2f" % Gy, u'\u00b0' + "/s", "\tGz=%.2f" % Gz, u'\u00b0' + "/s",
          "\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az)
    print("x rotation: ", get_x_rotation(acc_x, acc_y, acc_z))
    print("y rotation: ", get_y_rotation(acc_x, acc_y, acc_z))

    # a lo mejor aprovecho y meto en un .txt
    archive.write(f"ACC {Ax, Ay, Az}; GYRO {Gx, Gy, Gz}\n")  # ojala se entienda
    sleep(3)
