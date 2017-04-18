from BayesNet import *

def solveQueryWeight(samples, query, evidence):
    querySamples = getConsistentSamples(query, samples)
    numerator = getTotalWeight(querySamples, evidence)

    denominator = getTotalWeight(samples, evidence)

    if denominator == 0:
        return error()

    return numerator / denominator

def solveQuery(samples, query):
    querySamples = getConsistentSamples(query, samples)
    numerator = len(querySamples)

    denominator = len(samples)

    if denominator == 0:
        return error()

    return numerator / denominator

def error():
    print("ERROR: there is no sample consistent with the evidence, set a ",
          "higher number of samples")
    return 0

def splitQuery(expression):
    if '|' in expression:
        query, evidence = expression.split('|')
        query = splitByComa(query)
        evidence = splitByComa(evidence)
    else:
        query = splitByComa(expression)
        evidence = []

    return (query, evidence)

def splitByComa(string):
    if ',' in string:
        return string.split(',')
    else:
        return [string]

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

def getTotalWeight(samples, evidence):
    if not evidence:
        return len(samples)

    totalWeight = 0

    for sample in samples:
        sampleWeight = 1
        for desc, prob in sample:
            if desc in evidence:
                sampleWeight *= prob
        totalWeight += sampleWeight

    return totalWeight

def getSample(evidence):
    probc = probC(evidence)
    probs = probS(evidence, probc)
    probr = probR(evidence, probc)
    probw = probW(evidence, probs, probr)

    return [probc, probs, probr, probw]
