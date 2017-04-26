from Standars import *

def learn(data):
    featureSz = len(data[0]) - 1
    weights = [ [0 for j in range(featureSz)] for i in range(numClass)]

    learned = False

    while not learned:
        allgood = True

        for sample in data:
            realClass = sample[-1]
            calcClass = perceptron(sample, weights)

            if realClass != calcClass:
                allgood = False
                weights[realClass] = incrementWeight(sample, weights[realClass])
                weights[calcClass] = decrementWeight(sample, weights[calcClass])

        if allgood:
            learned = True

    return weights

def perceptron(sample, weights):
    first = True
    classification = 0

    for classVal,weight in enumerate(weights):
        ans = 0

        for i, val  in enumerate(weight):
            ans += sample[i] * val

        if first:
            maxVal = ans 
            first = False
        elif ans > maxVal:
            maxVal = ans    
            classification = classVal
        
    return classification

def incrementWeight(feature, weight):
    size = len(feature)

    for i in range(size-1):
        weight[i] += feature[i]

    return weight

def decrementWeight(feature, weight):
    size = len(feature)

    for i in range(size-1):
        weight[i] -= feature[i]

    return weight
