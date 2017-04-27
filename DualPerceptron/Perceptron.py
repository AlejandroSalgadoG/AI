from Standars import *

def learn(data):
    featureSz = len(data[0]) - 1 # dont count the classification
    alphas = [ [0 for j in range(featureSz)] for i in range(numClass)]

    learned = False

    while not learned:
        allgood = True

        for n, sample in enumerate(data):
            realClass = sample[-1]
            calcClass = perceptron(data, sample, alphas)

            if realClass != calcClass:
                allgood = False
                alphas[realClass][n] += 1
                alphas[calcClass][n] -= 1

        if allgood:
            learned = True

    return alphas

def perceptron(data, sample, alphas):
    first = True
    classification = 0

    for classVal,alphaArr in enumerate(alphas):
        ans = 0

        for idx, alpha  in enumerate(alphaArr):
            ans += alpha * K( data[idx], sample)

        if first:
            maxVal = ans 
            first = False
        elif ans > maxVal:
            maxVal = ans    
            classification = classVal
        
    return classification

def K(x1, x2):
    dot = 0
    size = len(x1) - 1

    for i in range(size):
        dot += x1[i] * x2[i]

    return dot

def K2(x1, x2):
    dot = 0
    size = len(x1) - 1

    for i in range(size):
        dot += x1[i] * x2[i]

    dot += 1

    return dot**2
