from graphics import *

from Standars import *

class Gui:

    win = None
    rectSize = size / numRow
    colors = []

    def __init__(self):
        self.win = GraphWin("Exact Hmm", size + btnSize, size)
        self.setBlackColors()
        self.drawGrid()
        self.drawEndBtn("black")
        self.drawNextBtn("black")

    def setBlackColors(self):
        self.colors = ["black" for i in range(numRow **2)]

    def drawEndBtn(self, color):
        p1 = Point(size, 0)
        p2 = Point(size+btnSize, size/2)

        rect = Rectangle(p1, p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")

        x,y = (size + size+btnSize)/2, size/4
        pos = Point(x,y)
        
        label = Text(pos, "BUST")
        label.setTextColor("white")
        label.draw(self.win)

    def drawNextBtn(self, color):
        p1 = Point(size, size/2)
        p2 = Point(size + btnSize, size)

        rect = Rectangle(p1, p2)
        rect.draw(self.win)
        rect.setFill(color)
        rect.setOutline("white")

        x,y = (size + size+btnSize)/2, (size*3)/4
        pos = Point(x,y)
        
        label = Text(pos, "NEXT")
        label.setTextColor("white")
        label.draw(self.win)

    def drawGrid(self):
        for i in range(numRow):
            for j in range(numRow):
                colorIdx = j + i*numRow
                color = self.colors[colorIdx]

                self.drawRect(i,j, color)

    def drawRect(self, i,j, color):                                              
        p1 = Point(j * self.rectSize, i * self.rectSize)                         
        p2 = Point(p1.x + self.rectSize, p1.y + self.rectSize)                   
                                                                                 
        rect = Rectangle(p1,p2)                                                  
        rect.draw(self.win)                                                      
                                                                                 
        rect.setFill(color)                                                      
        rect.setOutline("white")

    def drawProb(self, probs):
        self.drawGrid()

        for i in range(numRow):
            for j in range(numRow):
                pos = j + i*numRow
                strProb = "%.2f" % probs[pos]

                self.writeMsg(i,j, strProb, "blue")

    def getMouse(self):
        point = self.win.getMouse()

        j, i = point.x, point.y
        i = i // self.rectSize
        j = j // self.rectSize

        return (int(i), int(j))

    def writeMsg(self, i,j, msg, color):
        x,y = j*self.rectSize, i*self.rectSize
        halfRectSz = self.rectSize / 2

        pos = Point(x + halfRectSz, y + halfRectSz)
        label = Text(pos, msg)
        label.setTextColor(color)
        label.draw(self.win)

    def drawSensorReading(self, pos, color):
        i,j = pos
        pos = j+i*numRow
        self.colors[pos] = color

    def drawResult(self, pos, result):                                           
        if result:                                                               
            color,msg = "red", "HIT!"                                            
        else:                                                                    
            color,msg = "blue", "MISS!"                                          
                                                                                 
        i,j = pos                                                                
        self.drawRect(i,j, color)                                                
        self.writeMsg(i,j, msg, "white")
