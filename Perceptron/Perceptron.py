def learn(data):
    featureSz = len(data[0]) - 1
    weight = [ 0 for i in range(featureSz)]

    learned = False

    while not learned:
        allgood = True

        for sample in data:
            realClass = sample[-1]
            calcClass = perceptron(sample, weight, featureSz)

            if realClass != calcClass:
                allgood = False
                weight = adjustWeights(realClass, calcClass, sample, weight, featureSz)
                break

        if allgood:
            learned = True

    return weight

def perceptron(sample, weight, size):
    val = 0
    for i in range(size):
        val += sample[i] * weight[i]

    if val <= 0:
        return 0
    else:
        return 1

def adjustWeights(realClass, calcClass, sample, weight, size):
    diff = realClass - calcClass
    newWeigth = []

    for i in range(size):
        newVal = weight[i] + diff * sample[i]
        newWeigth.append(newVal)

    return newWeigth
