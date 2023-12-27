from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
from lora_reciever import Receiver
from plot_altitude import *

fig, ax = plt.subplots()

names = ['Temperature', 'Humidity', 'Pressure', 'Altitude']
boundary = [(0,50), (0,1000), (0,2000), (0,200)]

x = []
y = []
anim = FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y, 0))
plt.show()