#!/bin/python

from BayesNet import *

def main():
    probc = probC()
    probs = probS(probc)
    probr = probR(probc)
    probw = probW(probs, probr)
    
    print(probc, probs, probr, probw)

main()
