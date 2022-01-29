import numpy as np

MOVING_LIST = []

movingAV = 0.0
for i in range(1, 15):
    MOVING_LIST.append(i)
    if len(MOVING_LIST) <= 10:
        movingAV = np.sum(MOVING_LIST) / len(MOVING_LIST)
        print("Before 10 " , movingAV)
        continue

    movingAV = movingAV - (MOVING_LIST[0] / 10) + (i / 10)

    del MOVING_LIST[0]

    print("NUM",len(MOVING_LIST))
    print("List",MOVING_LIST)
    print("MAV", movingAV)

