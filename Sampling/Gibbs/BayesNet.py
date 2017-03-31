from __future__ import division

import random

from Distribution import *

def rand():
    return random.randint(0,100) / 100

def getEvidenceVar(var, evidence):
    evidenceDesc = [desc[1:] for desc in evidence]
    for idx, desc in enumerate(evidenceDesc):
        if desc == var:
            return evidence[idx]
    return None

def getElem(var, dist):
    for desc, prob in dist:
        if desc == var:
            return (desc, prob)

def selectElem(table):
    gen = rand()
    for desc, prob in table:
        gen -= prob
        if gen <= 0:
            return (desc, prob)

def probC(evidence):
    evidenceVar = getEvidenceVar("c", evidence)

    if evidenceVar is None:
        return selectElem(cloudy)
    else:
        return getElem(evidenceVar, cloudy)

def probR(evidence, probC):
    evidenceVar = getEvidenceVar("r", evidence)

    desc, prob = probC
    table = rain[desc]

    if evidenceVar is None:
        return selectElem(table)
    else:
        return getElem(evidenceVar, table)

def probS(evidence, probC):
    evidenceVar = getEvidenceVar("s", evidence)

    desc, prob = probC
    table = sprinkler[desc]

    if evidenceVar is None:
        return selectElem(table)
    else:
        return getElem(evidenceVar, table)

def probW(evidence, probS, probR):
    evidenceVar = getEvidenceVar("w", evidence)

    descS, probS = probS
    descR, probR = probR
    table = wetgrass[descS][descR]

    if evidenceVar is None:
        return selectElem(table)
    else:
        return getElem(evidenceVar, table)

def getRandVar(query):
    querySz = len(query)

    idx = random.randint(0,querySz-1)

    return query[idx]

def getOpElem(var, dist):
    for desc, prob in dist:
        if desc != var:
            return (desc,prob)

def getCConditioned(varDict, samples):
    c = varDict["c"]
    r = varDict["r"]
    s = varDict["s"]

    desc, probc = getElem(c, cloudy)
    null, probr = getElem(r, rain[c])
    null, probs = getElem(s, sprinkler[c])

    opDesc, compc = getOpElem(c, cloudy)
    null, compr = getOpElem(r, rain[c])
    null, comps = getOpElem(s, sprinkler[c])

    numerator = probc * probs * probr
    opNumerator = compc * comps * compr
    denominator = numerator + opNumerator

    probc = numerator / denominator
    compc = opNumerator / denominator

    return [(desc,probc),(opDesc,compc)]

def getRConditioned(varDict, samples):
    c = varDict["c"]
    r = varDict["r"]
    s = varDict["s"]
    w = varDict["w"]

    desc, probr = getElem(r, rain[c])
    null, probw = getElem(w, wetgrass[s][r])

    opDesc, compr = getOpElem(r, rain[c])
    null, compw = getOpElem(w, wetgrass[s][r])

    numerator = probr * probw
    opNumerator = compr * compw
    denominator = numerator + opNumerator

    probr = numerator / denominator
    compr = opNumerator / denominator

    return [(desc,probr),(opDesc,compr)]

def getSConditioned(varDict, samples):
    c = varDict["c"]
    r = varDict["r"]
    s = varDict["s"]
    w = varDict["w"]

    desc, probs = getElem(s, sprinkler[c])
    null, probw = getElem(w, wetgrass[s][r])

    opDesc, comps = getOpElem(s, sprinkler[c])
    null, compw = getOpElem(w, wetgrass[s][r])

    numerator = probs * probw
    opNumerator = comps * compw
    denominator = numerator + opNumerator

    probr = numerator / denominator
    compr = opNumerator / denominator

    return [(desc,probs),(opDesc,comps)]

def getWConditioned(varDict, samples):
    r = varDict["r"]
    s = varDict["s"]
    w = varDict["w"]

    desc, probw = getElem(w, wetgrass[s][r])

    opDesc, compw = getOpElem(w, wetgrass[s][r])

    numerator = probw
    opNumerator = compw
    denominator = numerator + opNumerator

    probr = numerator / denominator
    compr = opNumerator / denominator

    return [(desc,probw),(opDesc,compw)]
