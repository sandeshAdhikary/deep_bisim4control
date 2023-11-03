from time import sleep
from rich.progress import Progress


for idx in range(100):
    with Progress() as progress:
        for idy in range(200):
            print(idy)
            if idy == 100:
                raise ValueError
            sleep(0.02)
        print("HHHHERRREE")
