#!/bin/python

from Solver import *

def main(samplings, iterations, expression):
    query, evidence = splitQuery(expression)    
    varDict = createVarDict(query + evidence)

    samples = []

    while samplings > 0:
        sample = getSample(evidence)

        tmpIter = iterations

        while tmpIter > 0:
            var = getRandVar(query)

            newDesc, newProb = getNewVarVal(var, varDict, sample)
            sample = updateSample(newDesc, newProb, sample)
            
            tmpIter -= 1
    
        samples.append(sample)

        samplings -= 1

    prob = solveQuery(samples, query, evidence)
    print("P(%s) = %f" % (expression, prob) )

main(3, 5, "+c,+r|+s,+w")
