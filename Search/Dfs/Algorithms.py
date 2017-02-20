import time

from Parameters import numCol,numRow

closeSet = set()

def executePlayer(win, board, player):
    startState = player.pos
    dfs(win, board, startState, player)

def dfs(win, board, state, player):
    succ = successors(board.board, state)
    for state in succ:
        if goal(board.board, *state):
            board.movePlayer(win,*state, player)
            time.sleep(1)
            print("Done!")
            return True
        board.movePlayer(win,*state, player)
        time.sleep(1)
        if dfs(win, board, state, player):
            return True
    return False
        
def goal(board, x,y):
    if board[y][x] == 3:
        return True
    else:
        return False
    
def successors(board, state):

    if state in closeSet:
        return []

    closeSet.add(state)

    succ = []
    x, y = state
    
    if x+1 < numCol and board[y][x+1] != 1:
        right = (x+1,y)
        if right not in closeSet:
            succ.append( right ) 
    if y+1 < numRow and board[y+1][x] != 1:
        down = (x,y+1)
        if down not in closeSet:
            succ.append( down ) 
    if x-1 >= 0 and board[y][x-1] != 1:
        left = (x-1,y)
        if left not in closeSet:
            succ.append( left ) 
    if y-1 >= 0 and board[y-1][x] != 1:
        up = (x,y-1)
        if up not in closeSet:
            succ.append( up ) 
              
    return succ
