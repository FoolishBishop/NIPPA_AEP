import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TEMPORAL
import random


# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    # Read height (meters) from BME280
    height = random.randint(-5, 10)  # replace random for sensor value

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f')[:-3])
    ys.append(height)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('BME280 Height over Time')
    plt.ylabel('meters')


# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=500)
# interval ms
plt.show()
