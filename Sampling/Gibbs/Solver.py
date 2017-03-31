from Distribution import *
from BayesNet import *

def splitQuery(expression):
    expression = expression.split('|')
    query = expression[0].split(',')
    evidence = expression[1].split(',')
    return (query, evidence)

def getNewVarVal(var, sample):
    desc = var[1:]

    if desc == "c":
        table = getCConditioned(desc, sample)
    if desc == "s":
        table = getSConditioned(desc, sample)
    if desc == "r":
        table = getRConditioned(desc, sample)
    if desc == "w":
        table = getWConditioned(desc, sample)

    return selectElem(table)

def updateSample(newDesc, newProb, sample):
    tmpDesc = newDesc[1:]
    newSample = [desc[1:] for desc, prob in sample]

    for idx, desc in enumerate(newSample):
        if desc == tmpDesc:
            sample[idx] = (newDesc, newProb)

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
