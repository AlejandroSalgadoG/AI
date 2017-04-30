from GuiStandars import *
from Random import *

#P(col | pos)   g     y     o     r
model = { 0 : [0.05, 0.05, 0.05, 0.85 ],
          1 : [0.05, 0.05, 0.85, 0.05 ],
          2 : [0.05, 0.85, 0.05, 0.05 ],
          3 : [0.85, 0.05, 0.05, 0.05 ],
          4 : [0.85, 0.05, 0.05, 0.05 ] }

colors = { 0 : "green",
           1 : "yellow",
           2 : "orange",
           3 : "red",
           "green"  : 0,
           "yellow" : 1,
           "orange" : 2,
           "red"    : 3 }

def useSensor(pos, star):
    dist = getDistance(pos, star)

    if dist > maxDist:
        dist = maxDist

    table = model[dist]
    colorIdx = selectRandom(table)
    return colors[colorIdx]

def getDistance(pos1, pos2):
    i1,j1 = fromIdxToPos(pos1)
    i2,j2 = fromIdxToPos(pos2)
    return abs(i1-i2) + abs(j1-j2)

def fromIdxToPos(idx):
    idx -= 1
    i = idx // numRow
    j = idx - i * numRow
    return (i,j)
