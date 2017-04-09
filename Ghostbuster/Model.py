maxDist = 4

#P(col | pos)   g     y     o     r
model = { 0 : [0.05, 0.05, 0.05, 0.85 ],
          1 : [0.05, 0.05, 0.85, 0.05 ],
          2 : [0.05, 0.85, 0.05, 0.05 ],
          3 : [0.85, 0.05, 0.05, 0.05 ],
          4 : [0.85, 0.05, 0.05, 0.05 ] }

translation = { 0 : "green",
                1 : "yellow",
                2 : "orange",
                3 : "red",
                "green"  : 0,
                "yellow" : 1,
                "orange" : 2,
                "red"    : 3 }
