def solveQueryWeight(samples, query, evidence):
    join = query + evidence

    numeratorSamples = getConsistentSamples(join, samples)
    numerator = getTotalWeight(numeratorSamples)

    denominatorSamples = getConsistentSamples(evidence, samples)
    denominator = getTotalWeight(denominatorSamples)

    if denominator == 0:
        return error()

    return numerator / denominator

def solveQuery(samples, query, evidence):
    return 1

def error():
    print("ERROR: there is no sample consistent with the evidence, set a ",
          "higher number of samples")
    return 0

def splitQuery(expression):
    expression = expression.split('|')
    query = expression[0].split(',')
    evidence = expression[1].split(',')
    return (query, evidence)

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

def getTotalWeight(samples):
    totalWeight = 0

    for sample in samples:
        sampleWeight = 1
        for desc, prob in sample:
            sampleWeight *= prob
        totalWeight += sampleWeight

    return totalWeight
