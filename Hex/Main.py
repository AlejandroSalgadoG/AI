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

    board.drawMove(win, *move, player1)

    win.getMouse()

setup()
