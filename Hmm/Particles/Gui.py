from graphics import *
from Parameters import *

class Gui:

    win = None
    rectSize = size / numRow

    def __init__(self):
        self.win = GraphWin("Exact Hmm", size, size)
        
    def drawGrid(self):
        for i in range(numRow):
            for j in range(numRow):
                x = j*self.rectSize
                y = i*self.rectSize

                p1 = Point(x, y)
                p2 = Point(x + self.rectSize, y + self.rectSize)

                rect = Rectangle(p1,p2)
                rect.draw(self.win)
                rect.setFill("black")
                rect.setOutline("white")

    def drawProb(self, probs):
        for i in range(numRow):
            for j in range(numRow):
                x = j*self.rectSize
                y = i*self.rectSize
                halfRectSz = self.rectSize /2

                point = Point(x + halfRectSz, y + halfRectSz)
            
                pos = j + i*numRow 
                
                self.drawProbs(point, probs[pos])

    def getMouse(self):
        point = self.win.getMouse()

        j, i = point.x, point.y
        i = i // self.rectSize
        j = j // self.rectSize

        return (int(i), int(j))

    def drawProbs(self, pos, prob):
        x,y = pos.x, pos.y
        point = Point(x,y)
        strProb = "%.2f" % prob
        label = Text(point, strProb)
        label.setTextColor("blue")
        label.draw(self.win)

    def drawSensorReading(self, pos, color):
        i,j = pos
    
        x = j*self.rectSize
        y = i*self.rectSize

        p1 = Point(x, y)
        p2 = Point(x + self.rectSize, y + self.rectSize)

        rect = Rectangle(p1,p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")
