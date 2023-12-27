from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
from lora_reciever import Receiver

x = []
y = [[]*4]

fig, ax = plt.subplots(ncols = 1, nrows = 4)
names = ['Temperature', 'Humidity', 'Pressure', 'Altitude']
boundary = [(0,50), (0,1000), (0,2000), (0,200)]

def animate(i,x,y):
    receiver = Receiver()
    packet = None
    packet = receiver.receiver.receive()
    if packet is not None:
        packet = str(packet, 'utf-8')[1:-1]
        data = [float(values) for values in packet.split(',')][1:]

        for index in range(4):
            variable = data[index]
            
            x.append(dt.datetime.now().strftime('%H:%M:%S'))
            y[index].append(variable)

            x = x[-20:]
            y[index] = y[index][-20:]
            ax[index].clear()
            ax[index].plot(x,y[index])
            ax[index].set_title(f'{names[index]} over time')
            ax[index].set_ylim(boundary[index])

        plt.xticks(rotation=45)

FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,60)))