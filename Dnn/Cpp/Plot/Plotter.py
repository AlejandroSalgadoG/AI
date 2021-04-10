#!/bin/env python

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def data_gen(t=0):
    while True:
        t += 1
        val = float(input())
        print("epoch %d, error %f" % (t, val))
        yield t, val

def init():
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 10)
    line.set_data(epoch, error)
    return line

def run(data):
    x, y = data
    epoch.append(x)
    error.append(y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if x >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()

    if y >= ymax:
        ax.set_ylim(ymin, 2*ymax)
        ax.figure.canvas.draw()

    line.set_data(epoch, error)
    line.set_color('blue')

    return line

fig, ax = plt.subplots()
plt.title('Training process')
plt.ylabel('Error')
plt.xlabel('Epoch')

line, = ax.plot([], [])
epoch, error = [], []
ax.grid()

ani = animation.FuncAnimation(fig, run, data_gen, interval=1000, repeat=False, init_func=init)
plt.show()
