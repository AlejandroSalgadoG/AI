#!/bin/python

from Gui import *

from Random import getRandPos

from Agent import *
from AgentInterface import *
from Standars import *

def main():
    p1Gui = Gui("Player 1", 1)
    p2Gui = Gui("Player 2", 2)

    pos1 = getRandPos()
    pos2 = getRandPos()

    agent = Agent()

    opAction = [0,0,0]
    result2 = None

    p1Gui.drawStar(pos1)

    while True:
        print(pos2)

        type1, param1 = opponent(p1Gui, p2Gui)
        result1 = executeHumanAction(p1Gui, p2Gui, type1, param1, pos1, pos2)

        if type1 == move:
            pos1 = param1

        p1Gui.clean()
        p1Gui.drawStar(pos1)

        opAction[0] = type1
        opAction[1] = param1
        opAction[2] = result1

        type2, param2 = agent.play(2, result2, opAction, pos2)
        result2 = executeCpuAction(p1Gui, type2, param2, pos2, pos1)

        if type2 == move:
            pos2 = param2
        
main()
