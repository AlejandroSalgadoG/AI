#!/bin/python

from Parameters import *
from Gui import *
from Solver import *

def main():
    gui = Gui()

    generateGhost()
    iniProbs = getInitialDist()
    particles = distributeParticles(particleNum, iniProbs)
    probs = getProbs(particles)

    while True:
        gui.drawProb(probs)
        pos = gui.getMouse()

        if pos[1] > numRow-1:
            if pos[0] >= numRow/2:
                gui.setBlackColors()
                moveGhost()
                particles = moveParticles(particles)
                probs = getProbs(particles)
            else:
                gui.drawEndBtn("blue")
                pos = gui.getMouse()
                result = isGhostThere(pos)
                gui.drawResult(pos, result) 
                revealGhost()
                break
        else:
            color = useSensor(pos)
            gui.drawSensorReading(pos, color)
            condProbs = getNewPosDist(pos, color, probs)

            weights = weightParticles(particles, condProbs)
            normWeights = normalize(weights)
            particles = redistributeParticles(particles, normWeights)
            probs = getProbs(particles)

    gui.getMouse()
        
main()
