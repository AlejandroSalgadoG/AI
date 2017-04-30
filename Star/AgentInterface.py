from GuiStandars import *
from Standars import *
from Sensor import *

directions = {                                                                   
               "up"    : 1,                                                        
               "right" : 2,                                                        
               "down"  : 3,                                                        
               "left"  : 4,                                                        
               1 : "up",                                                           
               2 : "right",                                                        
               3 : "down",                                                         
               4 : "left"                                                          
             }

def getDirection(old, new):
    i1 = old // numRow
    j1 = old - i1 * numRow

    i2 = new // numRow
    j2 = new - i2 * numRow

    if i1 > i2:
        return up
    elif j1 < j2:
        return right
    elif i1 < i2:
        return down
    else:
        return left

def executeCpuAction(p2Gui, tipo, parametro, pos1, pos2):
    if tipo == move:
        direction = directions[parametro]
        print("player 1 move", direction)

        return getDirection(pos1, parametro) 

    elif tipo == sense:
        color = useSensor(parametro, pos2)
        p2Gui.drawSensorReading(parametro, color)

        return color

    elif tipo == shoot:
        if parametro == pos2:
            result = 1
        else:
            result = 0

        p2Gui.drawResult(parametro, result)
        
        return result
    else:
        print("!!!!!!!!!!!! invalid action !!!!!!!!!!!!!!!!!")

def executeHumanAction(p1Gui, p2Gui, tipo, parametro, pos1, pos2):
    if tipo == move:
        p2Gui.drawStar(parametro)
        return getDirection(pos2, parametro)

    elif tipo == sense:
        color = useSensor(parametro, pos1)
        p1Gui.drawSensorReading(parametro, color)

        return color

    elif tipo == shoot:
        if parametro == pos1:
            result = 1
        else:
            result = 0

        p1Gui.drawResult(parametro, result)

        return result

def opponent(p1Gui, p2Gui):
    btn = p2Gui.getMouse()

    action = buttonCode[btn]

    if action == "move":
        pos = p2Gui.getMouse()
        return move, pos

    elif action == "sense":
        pos = p1Gui.getMouse()
        return sense, pos

    elif action == "shoot":
        pos = p1Gui.getMouse()
        return shoot, pos
