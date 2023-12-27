from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
from lora_reciever import Receiver

x = []
y = []

fig, ax = plt.subplots(ncols = 1, nrows = 4)
names = ['Temperature', 'Humidity', 'Pressure', 'Altitude']
def animate(i,x,y, index: int, boundary: tuple):
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

        ax[index].clear()
        ax[index].plot(x,y)
        ax[index].set_title(f'{names[index]} over time')
        ax[index].set_ylim(boundary)

        plt.xticks(rotation=45)

ani = FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,60)))
ani = FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,1, (0,1000)))
ani = FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,20000)))
ani = FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,200)))

plt.show()