#!/bin/python

from Gui import Gui
from Model import *

def main():

    gui = Gui()
    gui.drawGrid()

    generateGhost()
    probs = getInitialPosDist()

    gui.drawProb(probs)

    while True:
        pos = gui.getMouse()
        color = useSensor(pos)

        probs = getNewPosDist(pos, color, probs)

        gui.drawSensorReading(pos, color)
        gui.drawProb(probs)

main()
