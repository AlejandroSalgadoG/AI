from graphics import *

from GuiStandars import *

class Gui:

    win = None
    player = 0

    def __init__(self, name, player):
        self.player = player
        self.win = GraphWin(name, size + btnSize, size)
        self.clean()
        self.setBtnSpace("black")
        self.drawMoveBtn("black")
        self.drawSenseBtn("black")
        self.drawShootBtn("black")

    def setBtnSpace(self, color):
        p1 = Point(size, 0)
        p2 = Point(size+btnSize, size)

        rect = Rectangle(p1, p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")

    def drawMoveBtn(self, color):
        p1 = Point(size, 0)
        p2 = Point(size+btnSize, rectSize)

        rect = Rectangle(p1, p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")

        x,y = (size + size+btnSize)/2, rectSize/2
        pos = Point(x,y)

        label = Text(pos, "Move")
        label.setTextColor("white")
        label.draw(self.win)

    def drawSenseBtn(self, color):
        p1 = Point(size, rectSize*2)
        p2 = Point(size + btnSize, rectSize*3)

        rect = Rectangle(p1, p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")

        x,y = (size + size+btnSize)/2, size/2
        pos = Point(x,y)

        label = Text(pos, "Sense")
        label.setTextColor("white")
        label.draw(self.win)

    def drawShootBtn(self, color):
        p1 = Point(size, rectSize*4)
        p2 = Point(size + btnSize, size)

        rect = Rectangle(p1, p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")

        x,y = (size + size+btnSize)/2, (rectSize*4 + size)/2
        pos = Point(x,y)

        label = Text(pos, "Shoot")
        label.setTextColor("white")
        label.draw(self.win)

    def clean(self):
        for pos in range(numRow**2):
            self.drawRect(pos, "black")

    def drawRect(self, pos, color):
        i,j = self.fromIdxToPos(pos)

        p1 = Point(j * rectSize, i * rectSize)
        p2 = Point(p1.x + rectSize, p1.y + rectSize)

        rect = Rectangle(p1,p2)
        rect.draw(self.win)

        rect.setFill(color)
        rect.setOutline("white")

    def drawProb(self, probs):
        self.clean()

        for pos in range(numRow**2):
            strProb = "%.2f" % probs[pos]
            self.writeMsg(pos, strProb, "blue")

    def getMouse(self):
        point = self.win.getMouse()

        j, i = point.x, point.y
        i = i // rectSize
        j = j // rectSize

        if j == numRow:
            return i + numRow**2 + 1

        return int(j + i*numRow) + 1

    def writeMsg(self, pos, msg, color):
        i,j = self.fromIdxToPos(pos)

        x,y = j*rectSize, i*rectSize
        halfRectSz = rectSize / 2

        pos = Point(x + halfRectSz, y + halfRectSz)
        label = Text(pos, msg)
        label.setTextColor(color)
        label.draw(self.win)

    def drawSensorReading(self, pos, color):
        pos -= 1
        self.drawRect(pos, color)

    def drawResult(self, pos, result):
        if result == 1:
            color,msg = "red", "HIT!"
        else:
            color,msg = "blue", "MISS!"

        pos -= 1

        self.drawRect(pos, color)
        self.writeMsg(pos, msg, "white")

    def fromIdxToPos(self, pos):
        i = pos // numRow
        j = pos - i * numRow
        return (i,j)

    def drawStar(self, pos):
        pos -= 1
        color = playerColor[self.player]
        self.drawRect(pos, color)
