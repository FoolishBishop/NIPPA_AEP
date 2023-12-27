from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
from lora_reciever import Receiver

fig, ax = plt.subplots()

names = ['Temperature', 'Humidity', 'Pressure', 'Altitude']
boundary = [(0,50), (0,1000), (0,2000), (0,200)]

def animate(i,x,y, index):
    receiver = Receiver()
    packet = None
    packet = receiver.receiver.receive()
    if packet is not None:
        packet = str(packet, 'utf-8')[1:-1]
        data = [float(values) for values in packet.split(',')][1:]

        variable = data[index]
        
        x.append(dt.datetime.now().strftime('%H:%M:%S'))
        y.append(variable)

        x = x[-20:]
        y = y[-20:]
        ax.clear()
        ax.plot(x,y[index])
        ax.set_title(f'{names[index]} over time')
        ax.set_ylim(boundary[index])

        plt.xticks(rotation=45)

x = []
y = []
anim = FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y, 3))
plt.show()