cloudy = [ ("+c", 0.5),
           ("-c", 0.5) ]

spklCplus = [ ("+s", 0.1),
              ("-s", 0.9) ]
spklCmin =  [ ("+s", 0.5),
              ("-s", 0.5) ]
sprinkler = { "+c" : spklCplus,
              "-c" : spklCmin }

rainCplus = [ ("+r", 0.8),
              ("-r", 0.2) ]
rainCmin =  [ ("+r", 0.2),
              ("-r", 0.8) ]
rain = { "+c" : rainCplus,
         "-c" : rainCmin }

wetSPlusPlus = [ ("+w", 0.99),
                 ("-w", 0.01) ]
wetSPlusMin =  [ ("+w", 0.90),
                 ("-w", 0.10) ]
wetSMinPlus =  [ ("+w", 0.90),
                 ("-w", 0.10) ]
wetSMinMin =   [ ("+w", 0.01),
                 ("-w", 0.99) ]
wetSPlus = { "+r" : wetSPlusPlus,
             "-r" : wetSPlusMin }
wetSMin =  { "+r" : wetSMinPlus,
             "-r" : wetSMinMin}
wetgrass = { "+s" : wetSPlus,
             "-s" : wetSMin }
