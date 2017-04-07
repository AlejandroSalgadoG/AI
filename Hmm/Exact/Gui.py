from graphics import *
from Parameters import *

class Gui:

    win = None
    rectSize = size / numRow
    colors = []

    def __init__(self):
        self.win = GraphWin("Exact Hmm", size, size)
        self.colors = ["black" for i in range(numRow **2)]
        self.drawGrid()

    def drawGrid(self):
        for i in range(numRow):
            for j in range(numRow):
                x = j*self.rectSize
                y = i*self.rectSize

                p1 = Point(x, y)
                p2 = Point(x + self.rectSize, y + self.rectSize)

                rect = Rectangle(p1,p2)
                rect.draw(self.win)

                color = self.colors[j+i*numRow]

                rect.setFill(color)
                rect.setOutline("white")

    def drawProb(self, probs):
        self.drawGrid()

        for i in range(numRow):
            for j in range(numRow):
                x = j*self.rectSize
                y = i*self.rectSize
                halfRectSz = self.rectSize / 2

                point = Point(x + halfRectSz, y + halfRectSz)

                pos = j + i*numRow

                self.writeProb(point, probs[pos])

    def writeProb(self, pos, prob):
        x,y = pos.x, pos.y
        point = Point(x,y)
        strProb = "%.2f" % prob
        label = Text(point, strProb)
        label.setTextColor("blue")
        label.draw(self.win)

    def getMouse(self):
        point = self.win.getMouse()

        j, i = point.x, point.y
        i = i // self.rectSize
        j = j // self.rectSize

        return (int(i), int(j))

    def drawSensorReading(self, pos, color):
        i,j = pos
        pos = j+i*numRow
        self.colors[pos] = color
