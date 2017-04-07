#!/bin/python

from Gui import *
from Solver import *

def main():
    gui = Gui()

    generateGhost()
    probs = getInitialDist()

    while True:
        gui.drawProb(probs)

        pos = gui.getMouse()
        color = useSensor(pos)

        gui.drawSensorReading(pos, color)

        moveGhost()

        probs = getNewDist(pos, color, probs)

main()
