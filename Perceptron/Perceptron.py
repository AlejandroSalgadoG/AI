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
                tao = calcTao(weights[realClass], weights[calcClass], sample)
                weights[realClass] = incrementWeight(sample, weights[realClass], tao)
                weights[calcClass] = decrementWeight(sample, weights[calcClass], tao)

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

def calcTao(goodW, badW, feature):
    size = len(goodW)
    diff = [0 for i in range(size)] 

    for i in range(size):
        diff[i] = badW[i] - goodW[i]

    dot = 0
    for i in range(size):
        dot += diff[i] * feature[i]

    num = dot + 1

    dot = 0
    for i in range(size):
        dot += feature[i] * feature[i]
    
    den = dot * 2

    return num/den

def incrementWeight(feature, weight, tao):
    size = len(feature)

    for i in range(size-1):
        weight[i] += tao * feature[i]

    return weight

def decrementWeight(feature, weight, tao):
    size = len(feature)

    for i in range(size-1):
        weight[i] -= tao * feature[i]

    return weight
