from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
from lora_reciever import Receiver

x = []
y = []

fig, ax = plt.subplots()

def animate(i,x,y):
    receiver = Receiver()
    packet = None
    packet = receiver.receiver.receive()
    if packet is not None:
        packet = str(packet, 'utf-8')[1:-1]
        data = [float(values) for values in packet.split(',')]
        altitude = data[3]
    
        x.append(dt.datetime.now().strftime('%H:%M:%S'))
        y.append(altitude)

        x = x[-20:]
        y = y[-20:]

        ax.clear()
        ax.plot(x,y)
        plt.xticks(rotation=45)

ani = FuncAnimation(fig, animate, interval=1000, repeat=False, fargs=(x,y))

plt.show()