import math

from Model import *

def mlSmooth(dataset, feature):
    probs = {}

    for data in dataset:
        if data[feature] in probs:
            probs[data[feature]] += 1
        else:
            probs[data[feature]] = 1
                
    size = len(dataset)

    for key in probs:
        probs[key] /= size

    return probs

def laplaceSmooth(dataset, k, feature):
    probs = {}

    for data in dataset:
        if data[feature] in probs:
            probs[data[feature]] += 1
        else:
            probs[data[feature]] = 1
                
    size = len(dataset)

    for key in probs:
        probs[key] /= size

    return probs

def naiveBayes(condProb, sample):
    probs = []
    for classId, key in enumerate(classesProb):
        prob = classesProb[key]
        pc = math.log(prob)

        featureTable = condProb[classId]

        pfc = 0
        for featureId, table in enumerate(featureTable):
            feature = sample[featureId]
            if feature in table:
                pfc += math.log(table[feature])
                
        probs.append(pc + pfc)

    return getMaxPos(probs)

def getMaxPos(probs):
    maximum = max(probs) 

    for idx, prob in enumerate(probs):
        if prob == maximum:
            return idx
