from Standars import *

def classify(data, sample):
    size = len(data)
    dot = similarity(data[0], sample)
    classification = [ (data[0], dot) ]

    for i in range(1,size):
        dot = similarity(data[i], sample)
        new = (data[i], dot)

        classification = insertInOrder(classification, new)

    nn = classification[-k:]

    return vote(nn)

def similarity(vec1, vec2):
    ans = 0
    for idx, vect2 in enumerate(vec2):
        ans +=  vec1[idx] * vect2

    return ans

def insertInOrder(classification, new):
    added = False

    for idx, sample in enumerate(classification):
        if new[1] <= sample[1]:
            classification.insert(idx, new) 
            added = True
            break

    if not added:
        classification.append(new) 

    return classification

def vote(samples):
    labels = {}

    for sample in samples:
        label = sample[0][-1]

        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1

    maxVal = 0
    for label in labels:
        if labels[label] > maxVal:
            classification = label
            maxVal = labels[label]

    return classification
