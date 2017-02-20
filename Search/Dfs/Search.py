from graphics import *

from Parameters import *
from Board import *
from Player import *
from Algorithms import *

boardReady = False
board = None
players = None

def setup():
    global board,players

    win = GraphWin("Search", widthSize, heightSize)
    win.setBackground("black")


    player1 = Player(2, p1Color, p1Pos)
    player2 = Player(3, p2Color, p2Pos)

    players = [player1, player2]

    board = Board(players)
    board.drawGrid(win, "white")

    board.drawPlayer(win, *p1Pos, player1)
    board.drawPlayer(win, *p2Pos, player2)

    while True:
        pos = win.getMouse()

        if win.checkKey():
            break

        pos = (pos.x//rectSize, pos.y//rectSize)
        board.createObstacle(win, *pos)

    board.drawGrid(win, "black")

    executePlayer(win, board, players[0])

    win.getMouse()

setup()
