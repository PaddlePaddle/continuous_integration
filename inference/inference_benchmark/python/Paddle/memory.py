from jtop import jtop
import time

if __name__ == "__main__":

    print("Simple Tegrastats reader")

    with jtop() as jetson:
        while True:
            # Read tegra stats
            print(jetson.stats)
            # Status disk
            print(jetson.disk)
            # Status fans
            if hasattr(jetson, 'fan'):
                print(jetson.fan)
            # uptim