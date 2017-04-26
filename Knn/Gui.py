import math

from graphics import *

from Standars import *

class Gui:

    win = None

    def __init__(self):
        self.win = GraphWin("Exact Hmm", sizeX, sizeY)
        self.win.setBackground("black")
        self.drawGrid()

    def drawGrid(self):
        for i in range(domain-1,0,-1):
            pv1 = Point(0, rectSize*i)
            pv2 = Point(sizeY, rectSize*i)

            ph1 = Point(rectSize*i, 0)
            ph2 = Point(rectSize*i, sizeY)

            vline = Line(pv1,pv2)
            vline.draw(self.win)

            hline = Line(ph1,ph2)
            hline.draw(self.win)
            
            vline.setFill("White")
            hline.setFill("White")

    def getMouse(self):
        point = self.win.getMouse()

        j, i = point.x, point.y
        i = i // rectSize
        j = j // rectSize

        return (int(i), int(j))

    def drawData(self, data):
        for x,y, color in data:
            p1 = self.fromCoorToPoint(x,y)

            point = Circle(p1,radius)

            point.draw(self.win)
            point.setFill(color)

    def fromCoorToPoint(self, x,y):
        return Point(rectSize*x, rectSize*(rangeSz-y))
