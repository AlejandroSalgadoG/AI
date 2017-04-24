from Parameters import *

size = numRow*100
btnSize = 50

def fromPosToIdx(pos):
    i,j = pos
    return j + i*numRow

def fromIdxToPos(idx):
    i = idx // numRow
    j = idx - i * numRow
    return (i,j)
