from graphics import *

from Parameters import *
from Board import *
from Player import *
from Agent import *

def setup():
    win = GraphWin("Hex", widthSize+2, heightSize+2)
    win.setBackground("black")

    player1 = Player(p1Id, p1Color)
    player2 = Player(p2Id, p2Color)

    players = [player1, player2]

    board = Board(players)
    board.drawGrid(win)

    move = agente(board.board, p1Id)

    board.drawMove(win, move[0], move[1], player1)

    while True:
        x,y = board.getMouse(win)

        if x < 0 or x > 10 or y < 0 or y > 10:
            continue
        else:
            board.drawMove(win, x, y, player2)

setup()
