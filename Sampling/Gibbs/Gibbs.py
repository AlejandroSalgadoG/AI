#!/bin/python

from BayesNet import *
from Solver import *

def main(samplings, iterations, expression):
    query, evidence = splitQuery(expression)    

    samples = []

    while samplings > 0:

        probc = probC()
        probs = probS(probc)
        probr = probR(probc)
        probw = probW(probs, probr)

        sample = ([probc, probs, probr, probw])
    
        while iterations > 0:
            var = getRandVar(query)

            newVal = getNewVarVal(var, sample)
            sample = updateSample(newVal, sample)
            
            iterations -= 1
    
        samples.append(sample)

        samplings -= 1

    prob = solveQuery(samples, query, evidence)
    print("P(%s) = %f" % (expression, prob) )

main(3, 5, "+c,+r|+s,+w")
