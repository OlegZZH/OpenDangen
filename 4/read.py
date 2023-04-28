import numpy as np
from matplotlib import pyplot as plt

f = open("Cur.txt", "r")
pxc=np.array(f.readline().split()).astype(float)
pyc=np.array(f.readline().split()).astype(float)
f.close()
# print(type(px[0]))






if __name__ == "__main__":
    fig2,ax2= plt.subplots()

    plt.gca().invert_yaxis()

    plt.gca().set_aspect("equal")

    curvaP = ax2.plot(pxc, pyc, linewidth=5, color="tab:orange")
    plt.show()