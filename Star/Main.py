#!/bin/python

from Standars import *
from Gui import *

from Sensor import *
from Random import getRandPos

from Agent import *

def getParam(old, new):
    i1 = old // numRow
    j1 = old - i * numRow 

    i2 = new // numRow
    j2 = new - i * numRow 

    if i1 > i2:
        return up
    elif j1 < j2:
        return right
    elif i1 < i2:
        return down
    else:
        return left

def main():
    player1Gui = Gui("Player 1", 1)
    player2Gui = Gui("Player 2", 2)

    pos1 = getRandPos()
    pos2 = getRandPos()

    print(pos1)
    player2Gui.drawStar(pos2)

    agent = Agent()

    opAction = [0,0,0]

    while True:
        tipo, parametro = agent.play(1, 0, opAction, pos1)

        if tipo == move:
            if parametro == up:
                print("player 1 move up")
            elif parametro == right:
                print("player 1 move right")
            elif parametro == down:
                print("player 1 move down")
            elif parametro == left:
                print("player 1 move left")
            else:
                print("!!!!!!!!!!!! invalid action !!!!!!!!!!!!!!!!!")

        elif tipo == sense:
            color = useSensor(parametro,pos2)
            player2Gui.drawSensorReading(parametro, color)

        elif tipo == shoot:
            if parametro == pos2:
                result = 1
            else:
                result = 0

            player2Gui.drawResult(parametro, result)

        else:
            print("!!!!!!!!!!!! invalid action !!!!!!!!!!!!!!!!!")
            
            

        btn = player2Gui.getMouse()

        action = buttonCode[btn]
        
        if action == "move":
            opAction[atype] = move

            oldPos = pos2
            pos2 = player2Gui.getMouse()
            player2Gui.drawStar(pos2)

            opAction[param] = getParam(oldPos, pos2)
            opAction[resul] = None

        elif action == "sense":
            opAction[atype] = sense

            pos = player1Gui.getMouse()
            color = useSensor(pos,pos1)
            player1Gui.drawSensorReading(pos, color)

            opAction[param] = pos
            opAction[resul] = color

        elif action == "shoot":
            opAction[atype] = shoot

            pos = player1Gui.getMouse()

            if pos == pos1:
                result = 1
            else:
                result = 0

            player1Gui.drawResult(pos, result)

            opAction[param] = pos
            opAction[resul] = result
main()
