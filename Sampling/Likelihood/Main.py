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

#    for sample in samples:
#        print(sample)
#    print()

    prob = solveQuery(samples, query)
    print("P(%s) = %f" % (expression, prob) )

    prob = solveQueryWeight(samples, query, evidence)
    print("WP(%s) = %f" % (expression, prob) )

main(500000, "+c,+r,+s,+w")
