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
    result1 = None

    while True:
        print(pos1)
        p2Gui.clean()
        p2Gui.drawStar(pos2)

        type1, param1 = agent.play(1, result1, opAction, pos1)
        result1 = executeCpuAction(p2Gui, type1, param1, pos1, pos2)

        if type1 == move:
            pos1 = param1

        type2, param2 = opponent(p1Gui, p2Gui)
        result2 = executeHumanAction(p1Gui, p2Gui, type2, param2, pos1, pos2)

        if type2 == move:
            pos2 = param2
        
        opAction[0] = type2
        opAction[1] = param2
        opAction[2] = result2

        
main()
