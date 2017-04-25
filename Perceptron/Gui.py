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

    def drawLine(self, weight):
        P1, P2 = weight

        if P1 == P2:
            return

        #Make sure that p2 its above p1
        if P1[1] > P2[1]:
            tmp = P1 
            P1 = P2
            P2 = tmp
        elif P1[1] == P2[1]:
            #If are at the same high p1 is the LHS point
            if P1[0] > P2[0]:
                tmp = P1 
                P1 = P2
                P2 = tmp

        p1 = self.fromCoorToPoint(P1[0], P1[1])
        p2 = self.fromCoorToPoint(P2[0], P2[1])

        line = Line(p1,p2)
        line.draw(self.win)
        
        line.setFill("Yellow")
        line.setWidth(thick)

        if P1[0] == P2[0]: # |
            dy = P2[1] - P1[1]
            p2 = self.fromCoorToPoint(P1[0]+dy, P1[1])

        elif P1[1] == P2[1]: # --
            dx = P2[0] - P1[0]
            p1 = self.fromCoorToPoint(P2[0], P2[1]+dx)

        elif P2[0] < P1[0]: #\
            dx = abs(P2[0] - P1[0])

            dx2 = (P2[0] - P1[0])**2
            dy2 = (P2[1] - P1[1])**2
            d = math.sqrt( dx2 + dy2 )

            p2 = self.fromCoorToPoint(P1[0]+d, P1[1]+dx)

        else: #/
            dx = abs(P2[0] - P1[0])

            dx2 = (P2[0] - P1[0])**2
            dy2 = (P2[1] - P1[1])**2
            d = math.sqrt( dx2 + dy2 )

            p2 = self.fromCoorToPoint(P1[0]-d, P1[1]+dx)

        
        
        line = Line(p1,p2)
        line.draw(self.win)
        
        line.setFill("Red")
        line.setWidth(thick)
