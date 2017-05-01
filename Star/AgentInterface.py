import sys

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

def executeCpuAction(otherGui, atype, param, ownPos, otherPos):
    if atype == move:
        direction = directions[param]
        print("player 1 move", direction)

        return None

    elif atype == sense:
        color = useSensor(param, otherPos)
        otherGui.drawSensorReading(param, color)

        return color

    elif atype == shoot:
        if param == otherPos:
            result = 1
        else:
            result = 0

        otherGui.drawResult(param, result)

        return result
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!! invalid action !!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

def executeHumanAction(ownGui, otherGui, atype, param, ownPos, otherPos):
    if atype == move:
        ownGui.drawStar(param)
        return None

    elif atype == sense:
        color = useSensor(param, otherPos)
        otherGui.drawSensorReading(param, color)

        return color

    elif atype == shoot:
        if param == otherPos:
            result = 1
        else:
            result = 0

        otherGui.drawResult(param, result)

        return result

def opponent(ownGui, otherGui):
    btn = ownGui.getMouse()

    action = buttonCode[btn]

    if action == "move":
        pos = ownGui.getMouse()
        return move, pos

    elif action == "sense":
        pos = otherGui.getMouse()
        return sense, pos

    elif action == "shoot":
        pos = otherGui.getMouse()
        return shoot, pos
