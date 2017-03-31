#!/bin/python

from Solver import *
from BayesNet import *

def main(iterations, expression):

    query, evidence = splitQuery(expression)

    samples = []

    while iterations > 0:
        sample = getSample(evidence)
        samples.append(sample)

        iterations -= 1

    prob = solveQuery(samples, query, evidence)
    print("P(%s) = %f" % (expression, prob) )

    prob = solveQueryWeight(samples, query, evidence)
    print("WP(%s) = %f" % (expression, prob) )

main(3, "+c,+r|+s,+w")
