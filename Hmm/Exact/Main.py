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
        

        if pos[1] > numRow-1:
            if pos[0] >= numRow/2:
                moveGhost()
            else:
                gui.drawEndBtn("blue")
                pos = gui.getMouse()
                result = isGhostThere(pos)
                gui.drawResult(pos, result) 
                revealGhost()
                break
            
#            color = useSensor(pos)
#            gui.drawSensorReading(pos, color)
#
#        if first:
#            probs = getNewDistBase(pos, color, probs)
#            first = False
#        else:
#            probs = getNewDistRec(pos, color, probs)

    gui.getMouse()

main()
