from Distribution import *

def splitQuery(expression):
    expression = expression.split('|')
    query = expression[0].split(',')
    evidence = expression[1].split(',')
    return (query, evidence)

def getNewVarVal(var, sample):
    return var

def updateSample(newVal, sample):
    return sample

def solveQuery(samples, query, evidence):
    numeratorSamples = getConsistentSamples(query, samples)
    numerator = len(numeratorSamples)
    
    denominator = len(samples)

    if denominator == 0:
        return error()

    return numerator / denominator

def getConsistentSamples(information, samples):
    consistentSamples = []
    
    infoSz = len(information)

    for sample in samples:
        consistent = 0
        for info in information:
            for desc, prob in sample:
                if info == desc:
                    consistent += 1 
                    break
        if consistent == infoSz:
            consistentSamples.append(sample)

    return consistentSamples
