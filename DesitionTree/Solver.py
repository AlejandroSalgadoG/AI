import math

def desitionTree(table):
    samples = len(table[0])
    featureSz = len(table) - 1

    information = []

    for i in range(featureSz):
        feature = table[i]
        for feat in feature:
#            entr = entropy(probs)
#            info = entropy * (count/samples) 
#            information.append(info)
        
#    feature = selectFeature(information)

    return 0    

def entropy(probs):
    acum = 0
    for prob in probs:
        acum += ( prob * math.log(prob,2) ) * -1
    return acum

def selectFeature(information):
    size = len(information) - 1
    minimum = information[-1] 
    feature = size

    for i in range(size):
        if information[i] < minimum:
            minimum = information[i]
            feature = i

    return feature
