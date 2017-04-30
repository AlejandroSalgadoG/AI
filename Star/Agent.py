class Agent:

    shoot = 1
    sense = 2
    move  = 3
    
    atype = 0
    param = 1
    resul = 2
    
    up    = 1
    right = 2
    down  = 3
    left  = 4

    def __init__(self):
        print("I emerge \o/")

    def play(self, jugador, resultado_accion, accion_oponente, estrellita):
        return self.shoot,3
