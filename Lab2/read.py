import numpy as np

from main import *
from matplotlib import pyplot as plt
# from Lab1 import PointBuilder



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
    fig2,ax2= plt.subplots()
    # img = plt.imread("palma.png")
    # img = plt.imread("monkey.jpg")
    # ax2.imshow(img)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    plt.gca().invert_yaxis()

    plt.gca().set_aspect("equal")
    # curvaM=ax2.plot(mxc,myc, linewidth=5, color="tab:orange")
    # lineM=ax2.plot(mx,my,linewidth=1, color="tab:blue")
    # pointM=ax2.plot(mx, my,'ro', picker=True, pickradius=5)

    t1=[0,1,2,3,4,5]
    t2=np.zeros_like(t1)
    curvaP = ax2.plot(pxc, pyc, linewidth=5, color="tab:orange")
    lineP = ax2.plot(t1, t2, linewidth=1, color="tab:blue")
    pointP = ax2.plot(t1, t2, 'ro', picker=True, pickradius=5)
    pointbuilder = PointBuilder(pointP[0], lineP[0], curvaP[0])
    pointbuilder.connect()
    # dr = DraggableRectangle(point[0])
    # dr.connect()
    # pointbuilder = PointBuilder(point[0], line[0], curva[0])
    # pointbuilder.connect()
    plt.show()