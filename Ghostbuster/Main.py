#!/bin/python

from Gui import Gui
from Solver import *

def main():

    gui = Gui()

    generateGhost()
    probs = getInitialPosDist()

    while True:
        gui.drawProb(probs)

        pos = gui.getMouse()
        
        if pos[1] > numRow-1:
            gui.drawEndBtn("blue")
            pos = gui.getMouse()
            result = isGhostThere(pos)
            gui.drawResult(pos, result) 
            revealGhost()
            break
        else: 
            color = useSensor(pos)
            gui.drawSensorReading(pos, color)
            probs = getNewPosDist(pos, color, probs)

    gui.getMouse()

main()
