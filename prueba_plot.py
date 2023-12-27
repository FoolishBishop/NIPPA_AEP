from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
from lora_reciever import Receiver
from multiprocessing import Process
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

if __name__ == '__main__':
    # Create processes for each animation
    processes = [
        Process(target=FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,60)))),
        Process(target=FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,1, (0,1000)))),
        Process(target=FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,20000)))),
        Process(target=FuncAnimation(fig, animate, interval=100, repeat=False, fargs=(x,y,0, (0,200))))
    ]

    # Start each process
    for process in processes:
        process.start()

    # Join all processes to ensure they complete before exiting
    for process in processes:
        process.join()

    # Display any necessary plots or visuals after animations are done
    plt.show()
