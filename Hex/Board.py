import sys

from graphics import *
from Parameters import *

widthSize = numCol * rectSize
heightSize = numRow * rectSize

class Board:
    board = None
    players = None

    def __init__(self, players):
        self.board = [[0 for i in range(numRow)] for j in range(numRow)]
        self.players = players

    def drawMove(self, win, x, y, player):

        self.board[y][x] = player.id

        x = x*rectSize + y*(rectSize/2)
        y = y*rectSize

        p1 = Point(x, y)
        p2 = Point(x+rectSize,y+rectSize)

        rect = Rectangle(p1,p2)
        rect.draw(win)
        rect.setFill(player.color)
        rect.setOutline("white")

    def drawGrid(self, win):
        for i in range(numRow):
            for j in range(numRow):
                x = j*rectSize + i*(rectSize/2)
                y = i*rectSize

                p1 = Point(x, y)
                p2 = Point(x+rectSize,y+rectSize)

                rect = Rectangle(p1,p2)
                rect.draw(win)
                rect.setFill("black")
                rect.setOutline("white")

    def getMouse(self, win):
        point = win.getMouse()

        x, y = point.x, point.y
        y = y//rectSize
        x -= (rectSize/2)*y
        x = x//rectSize

        return int(x), int(y)

    def printBoard(self):
        for i in range(numRow):
            for j in range(numRow):
                sys.stdout.write( str(self.board[i][j]) + " " )
            print("")
