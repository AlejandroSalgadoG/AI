#!/bin/python

from Gui import *
from Solver import *

def main(particleNum):
    gui = Gui()

    generateGhost()
    iniProbs = getInitialDist()
    particles = getParticles(particleNum, iniProbs)
    probs = getProbs(particles)

    gui.drawProb(probs)

    while True:
        pos = gui.getMouse()
        color = useSensor(pos)

        gui.drawSensorReading(pos, color)

        weight = getParticlesWeight(pos, color)
        particles = redistributeParticles(particleNum, weight)
        particles = moveParticles(particles)
        probs = getProbs(particles)

        gui.drawProb(probs)

        moveGhost()

main(1000)
