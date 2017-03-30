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

    prob = solveQuery(samples, query, evidence)
    print("P(%s) = %f" % (expression, prob) )

    prob = solveQueryWeight(samples, query, evidence)
    print("WP(%s) = %f" % (expression, prob) )

main(3, "+c,+r|+s,+w")
