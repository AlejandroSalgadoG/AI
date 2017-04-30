#!/bin/python

from Standars import *
from Gui import *

from Sensor import *
from Random import getRandPos

from Agent import *

def main():
    player1Gui = Gui("Player 1", 1)
    player2Gui = Gui("Player 2", 2)

    pos1 = getRandPos()
    pos2 = getRandPos()

    print(pos1)
    player2Gui.drawStar(pos2)

    agent = Agent()

    while True:
        tipo, parametro = agent.play(1, 0, [1, 1, 1], pos1)

        btn = player2Gui.getMouse()

        action = buttonCode[btn]
        
        if action == "move":
            pos2 = player2Gui.getMouse()
            player2Gui.drawStar(pos2)
        elif action == "sense":
            pos = player1Gui.getMouse()
            color = useSensor(pos,pos1)
            player1Gui.drawSensorReading(pos, color)
        elif action == "shoot":
            pos = player1Gui.getMouse()
            player1Gui.drawResult(pos, pos1)

main()
