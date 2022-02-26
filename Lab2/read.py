import numpy as np

from main import *
from matplotlib import pyplot as plt
from Lab1 import PointBuilder



f = open("PointU.txt", "r")
pxU = np.array(f.readline().split()).astype(float)
pyU = np.array(f.readline().split()).astype(float)
f.close()

f = open("CurU.txt", "r")
pxcU = np.array(f.readline().split()).astype(float)
pycU = np.array(f.readline().split()).astype(float)
f.close()

f = open("PointV.txt", "r")
pxV = np.array(f.readline().split()).astype(float)
pyV = np.array(f.readline().split()).astype(float)
f.close()

f = open("CurV.txt", "r")
pxcV = np.array(f.readline().split()).astype(float)
pycV = np.array(f.readline().split()).astype(float)
f.close()

# print(type(px[0]))

# f = open("TwitCur.txt", "r")
# mxc=np.array(f.readline().split()).astype(float)
# myc=np.array(f.readline().split()).astype(float)
# f.close()
#
# f = open("TwitPoint.txt", "r")
# mx=np.array(f.readline().split()).astype(float)
# my=np.array(f.readline().split()).astype(float)
# f.close()
# px[0]=px[-1]
# py[0]=py[-1]
# mx[0]=mx[-1]
# my[0]=my[-1]
# for i in range(len(px)-len(mx)):
#     mx=np.append(mx,mx[0])
#     my = np.append(my, my[0])
# print(len(mx))
# print(len(px))
# mx+=120
# mx*=1.4
# my*=1.4
if __name__ == "__main__":
    # paly=np.array(f.readline())
    fig,(ax1,ax2)= plt.subplots(1,2)
    # img = plt.imread("palma.png")
    # img = plt.imread("monkey.jpg")
    # ax2.imshow(img)
    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-0.1, 0.1)
    ax2.set_xlim(-0.1, 0.1)
    ax2.set_ylim(-0.1, 0.1)
    plt.gca().invert_yaxis()

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    # curvaM=ax2.plot(mxc,myc, linewidth=5, color="tab:orange")
    # lineM=ax2.plot(mx,my,linewidth=1, color="tab:blue")
    # pointM=ax2.plot(mx, my,'ro', picker=True, pickradius=5)
    curvaV = ax1.plot(pxcV, pycV, linewidth=5, color="tab:orange")
    lineV = ax1.plot(pxV, pyV, linewidth=1, color="tab:blue")
    pointV = ax1.plot(pxV, pyV, 'ro', picker=True, pickradius=5)
    pointbuilderV = PointBuilder(pointV[0], lineV[0], curvaV[0])
    pointbuilderV.connect()

    curvaU = ax2.plot(pxcU, pycU, linewidth=5, color="tab:orange")
    lineU = ax2.plot(pxU, pyU, linewidth=1, color="tab:blue")
    pointU = ax2.plot(pxU, pyU, 'ro', picker=True, pickradius=5)
    pointbuilderU = PointBuilder(pointU[0], lineU[0], curvaU[0])
    pointbuilderU.connect()
    # dr = DraggableRectangle(point[0])
    # dr.connect()
    # pointbuilder = PointBuilder(point[0], line[0], curva[0])
    # pointbuilder.connect()
    plt.show()