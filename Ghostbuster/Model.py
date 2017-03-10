import random

def rand():
    return random.randint(1,10)

def sensorModel(x, y):
    prob = rand()
    if x <= 1:
        if y <= 1:
            if prob <= 7:
                return (0.7, "R")
            if prob == 8:
                return (0.1, "O")
            if prob == 9:
                return (0.1, "Y")
            if prob == 10:
                return (0.1, "G")
        else:
            if prob <= 6:
                return (0.6, "R")
            if prob == 7 or prob == 8:
                return (0.2, "O")
            if prob == 9:
                return (0.1, "Y")
            if prob == 10:
                return (0.1, "G")
    else:
        if y <= 1:
            if prob <= 5:
                return (0.5, "R")
            if prob >= 6 or prob <= 8:
                return (0.3, "O")
            if prob == 9:
                return (0.1, "Y")
            if prob == 10:
                return (0.1, "G")
        else:
            if prob <= 4:
                return (0.4, "R")
            if prob >= 5 or prob <= 7:
                return (0.3, "O")
            if prob == 8 or prob == 9:
                return (0.2, "Y")
            if prob == 10:
                return (0.1, "G")
