#!/bin/python

from BayesNet import *
from Solver import *

def main(iterations, expression):

    query, evidence = splitQuery(expression)

    samples = []

    while iterations > 0:
        probc = probC()
        probs = probS(probc)
        probr = probR(probc)
        probw = probW(probs, probr)
        
        samples.append([probc, probs, probr, probw])

        iterations -= 1

    solveQuery(samples, query, evidence)
    solveQueryWeight(samples, query, evidence)

main(3, "+a,+r|+s,+w")
