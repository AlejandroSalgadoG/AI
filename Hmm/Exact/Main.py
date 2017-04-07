#!/bin/python

from Gui import *
from Solver import *

def main():
    gui = Gui()

    generateGhost()
    probs = getInitialDist()
    gui.drawProb(probs)

    first = True

    while True:
        pos = gui.getMouse()
        color = useSensor(pos)

        gui.drawSensorReading(pos, color)

        if first:
            probs = getNewDistBase(pos, color, probs)
            first = False
        else:
            probs = getNewDistRec(pos, color, probs)

        gui.drawProb(probs)

        moveGhost()

main()
