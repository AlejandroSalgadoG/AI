#P(col | pos)   g     y     o     r
model = { 0 : [0.4,  0.3,  0.2,  0.1  ],
          1 : [0.3,  0.4,  0.2,  0.1  ],
          2 : [0.2,  0.5,  0.2,  0.1  ],
          3 : [0.15, 0.25, 0.4,  0.2  ],
          4 : [0.1,  0.2,  0.25, 0.45 ] }

translation = { 0 : "green",
                1 : "yellow",
                2 : "orange",
                3 : "red",
                "green"  : 0,
                "yellow" : 1,
                "orange" : 2,
                "red"    : 3 }
