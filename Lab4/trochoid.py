import matplotlib.pyplot as plt
import numpy as np

def plot_trochoid(r=2,R=10,h=1):
    m=r/R
    t = np.linspace(0, 2*R * np.pi,100)
    x=(R+m*R)*np.cos(m*t)-h*np.cos(t+m*t)
    y=(R+m*R)*np.sin(m*t)-h*np.sin(t+m*t)
    fig,ax=plt.subplots()
    ax.plot(x,y)
    plt.axis('off')
    ax.set_aspect('equal')
    fig.savefig('demo.png', transparent=True,bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    plot_trochoid()