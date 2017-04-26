import time

def learn(data):
    featureSz = len(data[0]) - 1
    weight = [ 0 for i in range(featureSz) ]

    learned = False

    while not learned:
        allgood = True

        for sample in data:
            realClass = sample[-1]
            calcClass = perceptron(sample, weight, featureSz)

            if realClass != calcClass:
                allgood = False
                if calcClass == "red":
                    weight = adjustWeights("+", sample, weight, featureSz)
                else:
                    weight = adjustWeights("-", sample, weight, featureSz)
                break


        if allgood:
            learned = True

    return weight

def perceptron(sample, weight, size):
    val = 0
    for i in range(size):
        val += sample[i] * weight[i]

    if val <= 0:
        return "red"
    else:
        return "blue"

def adjustWeights(operation, sample, weight, size):
    newWeigth = []

    if operation == "+":
        for i in range(size):
            newVal = weight[i] + sample[i]
            newWeigth.append(newVal)
    else:
        for i in range(size):
            newVal = weight[i] - sample[i]
            newWeigth.append(newVal)

    return newWeigth
