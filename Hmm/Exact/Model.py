from Standars import *

maxDist = 4

#P(col | pos)   g     y     o     r
model = { 0 : [0.05, 0.05, 0.05, 0.85 ],
          1 : [0.05, 0.05, 0.85, 0.05 ],
          2 : [0.05, 0.85, 0.05, 0.05 ],
          3 : [0.85, 0.05, 0.05, 0.05 ],
          4 : [0.85, 0.05, 0.05, 0.05 ] }

def transition(pos):
    nr = numRow-1
    i,j = pos
    probs = [0.00 for i in range(numRow**2)]

    if i == 0: #upper
        if j == 0: #left corner

            probs[0] = 0.25 #stay the same
            idx = fromPosToIdx((0,1))
            probs[idx] = 0.50 # move right
            idx = fromPosToIdx((1,0))
            probs[idx] = 0.25 # move down

        elif j == nr:#right corner

            idx = fromPosToIdx((0,nr))
            probs[idx] = 0.25 #stay the same
            idx = fromPosToIdx((1,nr))
            probs[idx] = 0.50 # move down
            idx = fromPosToIdx((0,nr-1))
            probs[idx] = 0.25 # move left

        else:  #edge 

            idx = fromPosToIdx((0,j))
            probs[idx] = 0.16 #stay the same
            idx = fromPosToIdx((0,j+1))
            probs[idx] = 0.52 # move left
            idx = fromPosToIdx((1,j))
            probs[idx] = 0.16 # move down 
            idx = fromPosToIdx((0,j-1))
            probs[idx] = 0.16 # move right

    elif i == nr: #lower
        if j == 0: #left corner

            idx = fromPosToIdx((nr,0))
            probs[idx] = 0.25 #stay the same
            idx = fromPosToIdx((nr-1,0))
            probs[idx] = 0.50 # move up
            idx = fromPosToIdx((nr,1))
            probs[idx] = 0.25 # move right

        elif j == nr:#right corner

            idx = fromPosToIdx((nr,nr))
            probs[idx] = 0.25 #stay the same
            idx = fromPosToIdx((nr,nr-1))
            probs[idx] = 0.50 # move left
            idx = fromPosToIdx((nr-1,nr))
            probs[idx] = 0.25 # move up

        else:  #edge 

            idx = fromPosToIdx((nr,j))
            probs[idx] = 0.16 #stay the same
            idx = fromPosToIdx((nr,j-1))
            probs[idx] = 0.52 # move left
            idx = fromPosToIdx((nr-1,j))
            probs[idx] = 0.16 # move up
            idx = fromPosToIdx((nr,j+1))
            probs[idx] = 0.16 # move right

    else: #middle
        if j == 0: #left corner

            idx = fromPosToIdx((i,0))
            probs[idx] = 0.16 #stay the same
            idx = fromPosToIdx((i-1,0))
            probs[idx] = 0.52 # move up
            idx = fromPosToIdx((i,1))
            probs[idx] = 0.16 # move left
            idx = fromPosToIdx((i+1,0))
            probs[idx] = 0.16 # move down

        elif j == nr:#right corner

            idx = fromPosToIdx((i,nr))
            probs[idx] = 0.16 #stay the same
            idx = fromPosToIdx((i+1,nr))
            probs[idx] = 0.52 # move down
            idx = fromPosToIdx((i,nr-1))
            probs[idx] = 0.16 # move left
            idx = fromPosToIdx((i-1,nr))
            probs[idx] = 0.16 # move down

        else:  # center 

            idx = fromPosToIdx((i,j))
            probs[idx] = 0.12 # stay the same
            idx = fromPosToIdx((i,j+1))
            probs[idx] = 0.52 # move left
            idx = fromPosToIdx((i+1,j))
            probs[idx] = 0.12 # move down
            idx = fromPosToIdx((i,j-1))
            probs[idx] = 0.12 # move left
            idx = fromPosToIdx((i-1,j))
            probs[idx] = 0.12 # move up

    return probs

translation = { 0 : "green",
                1 : "yellow",
                2 : "orange",
                3 : "red",
                "green"  : 0,
                "yellow" : 1,
                "orange" : 2,
                "red"    : 3 }
