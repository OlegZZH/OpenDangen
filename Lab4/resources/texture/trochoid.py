import matplotlib.pyplot as plt
import numpy as np

def plot_trochoid(r=1,R=8,h=2):
    m=r/R
    t = np.linspace(0, 2*R * np.pi,500)
    x=(R+m*R)*np.cos(m*t)-h*np.cos(t+m*t)
    y=(R+m*R)*np.sin(m*t)-h*np.sin(t+m*t)
    fig,ax=plt.subplots()
    ax.plot(x,y, linewidth=7)
    plt.axis('off')
    ax.set_aspect('equal')
    fig.savefig('demo.png', transparent=True,bbox_inches='tight',dpi=500)
    plt.show()



if __name__ == '__main__':
    plot_trochoid()