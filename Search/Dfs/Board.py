import sys

from graphics import *
from Parameters import *

class Board:
    board = None
    players = None
    
    def __init__(self, players):
        self.board = [[0 for i in range(numCol)] for j in range(numRow)]
        self.players = players
        
    def createObstacle(self, win, x, y):
        x, y = int(x), int(y)

        if self.board[y][x] == 1:
            self.board[y][x] = 0
            color = "black"
        elif self.board[y][x] == 0:
            self.board[y][x] = 1
            color = "blue"
        else:
            return

        x,y = x*rectSize, y*rectSize
        rect = Rectangle(Point(x,y), Point(x+rectSize, y+rectSize))
        rect.draw(win)
        rect.setFill(color)
        rect.setOutline("white")

    def clearPos(self, win, x, y):
        self.board[y][x] = 0

        x,y = x*rectSize, y*rectSize
        rect = Rectangle(Point(x,y), Point(x+rectSize, y+rectSize))
        rect.draw(win)
        rect.setFill("black")
        rect.setOutline("black")
        
        
    def drawPlayer(self, win, x, y, player):
        radius = rectSize/2
        circle = Circle(Point(x*rectSize+radius, y*rectSize+radius), radius)
        circle.draw(win)
        circle.setFill(player.color)
        
        self.board[y][x] = player.id

    def drawGrid(self, win, color):
        for i in range(numCol):
            hline = Line(Point(0, i*rectSize), Point(widthSize, i*rectSize))
            vline = Line(Point(i*rectSize,0), Point(i*rectSize,heightSize))
            hline.draw(win)
            vline.draw(win)
            hline.setFill(color)
            vline.setFill(color)
        
    def printBoard(self):
        for i in range(numRow):
            for j in range(numCol):
                sys.stdout.write( str(self.board[i][j]) + " " )
            print("")
            
    def movePlayer(self, win, x, y, player):
        self.clearPos(win, *player.pos)
        player.pos = (x,y) 
        self.drawPlayer(win, x,y, player)
        
