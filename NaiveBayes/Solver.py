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
