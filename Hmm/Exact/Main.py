#!/bin/python

from Gui import *
from Solver import *

def main():
    gui = Gui()

    generateGhost()
    probs = getInitialDist()

    first = True

    while True:
        gui.drawProb(probs)

        pos = gui.getMouse()
        color = useSensor(pos)

        gui.drawSensorReading(pos, color)

        moveGhost()

        if first:
            probs = getNewDistBase(pos, color, probs)
            first = False
        else:
            probs = getNewDistRec(pos, color, probs)

main()
