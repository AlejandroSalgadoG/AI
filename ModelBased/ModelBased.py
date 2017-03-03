import random

a,b,c,d,e,x = 'a','b','c','d','e','x'

Ta,Tb,Tc,Td,Te = {}, {}, {}, {}, {}
R2 = {}
V2 = {}

def rand():
    return random.randint(1, 10)

def R(s,s2):
    if s == a:
        return -10
    elif s == d:
        return 10 
    else:
        return -1

def T(s):
    if s == a:
        return x
    if s == b:
        return c
    if s == c:
        prob = rand()
        if prob == 1:
            return a
        if prob == 2:
            return e
        else:
            return d
    if s == d:
        return x
    if s == e:
        return c

def storeT(s,s2):
    if s == a:
        Ta[s2] += 1
    if s == b:
        Tb[s2] += 1
    if s == c:
        Tc[s2] += 1
    if s == d:
        Td[s2] += 1
    if s == e:
        Te[s2] += 1

def storeV(episode):
    for idx, val in enumerate(episode):
        for tran in episode[idx:]:
            s = tran[0]
            s2 = tran[1]
            V2[val[0]] += R(s,s2)

def main():
    V2[a],V2[b],V2[c],V2[d],V2[e] = 0,0,0,0,0

    R2[b+c] = 0 
    R2[e+c] = 0
    R2[c+d] = 0
    R2[c+a] = 0
    R2[c+e] = 0
    R2[a+x] = 0
    R2[d+x] = 0

    Ta[x] = 0
    Tb[c] = 0
    Tc[a],Tc[d],Tc[e] = 0,0,0
    Td[x] = 0
    Te[c] = 0

    iterations = 800
    
    for i in range(iterations):
        if rand() < 6:
            s = b
        else:
            s = e

        episode = []
    
        while(s != x):
            s2 = T(s)
            storeT(s,s2)
            R2[s+s2] += R(s,s2)
            episode.append(s+s2)
            s = s2
        storeV(episode)

    totala = Ta[x]
    totalb = Tb[c]
    totalc = Tc[a] + Tc[d] + Tc[e]
    totald = Td[x]
    totale = Te[c]

    print("T(a,X,x) =",Ta[x] / totala)
    print("T(b,E,c) =",Tb[c] / totalb)
    print("T(c,E,a) =",Tc[a] / totalc)
    print("T(c,E,d) =",Tc[d] / totalc)
    print("T(c,E,e) =",Tc[e] / totalc)
    print("T(d,X,x) =",Td[x] / totald)
    print("T(e,N,c) =",Te[c] / totale)

    print("")

    print("R(b,E,c) =",R2[b+c] / Tb[c])
    print("R(e,E,c) =",R2[e+c] / Te[c])
    print("R(c,E,e) =",R2[c+e] / Tc[e])
    print("R(c,E,d) =",R2[c+d] / Tc[d])
    print("R(c,E,a) =",R2[c+a] / Tc[a])
    print("R(a,X,x) =",R2[a+x] / Ta[x])
    print("R(d,X,x) =",R2[d+x] / Td[x])

    print("")

    print("V(a) =",V2[a] / totala, "-", totala)
    print("V(b) =",V2[b] / totalb, "-", totalb)
    print("V(c) =",V2[c] / totalc, "-", totalc)
    print("V(d) =",V2[d] / totald, "-", totald)
    print("V(e) =",V2[e] / totale, "-", totale)

main()
