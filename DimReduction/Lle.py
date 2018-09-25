from Dataset import build_swiss_roll
from Plot3d import plot3d

def main():
    data, colors = build_swiss_roll(100)
    plot3d(data, colors)

if __name__ == '__main__':
   main()
