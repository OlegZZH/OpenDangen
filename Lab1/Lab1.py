import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})






def biz5(ra, rb, rc, rd, re, rf):
    t = np.linspace(0, 1, 50)
    r = []
    for u in t:
        r = np.append(r, ra * (1 - u) ** 5 + 5 * u * rb * (1 - u) ** 4 + (10 * rc * u ** 2) * (
                    1 - u) ** 3 + (10 * rd * u ** 3)*(1-u)**2+(5*re*u**4)*(1-u)+rf*u**5)

    return r


class PointBuilder():
    def __init__(self, point, line, curva):
        self.point = point
        self.press = None
        self.line = line
        self.curva = curva

    def connect(self):
        print("connect")
        self.cidpik = self.point.figure.canvas.mpl_connect('pick_event', self.onpick)
        self.cid = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)

    j = 0
    f = 0

    def on_press(self, event):
        print("on_press")
        if event.button == 1:
            x = np.array(self.point.get_data()[0])
            x = np.append(x, event.xdata)
            y = np.array(self.point.get_data()[1])
            y = np.append(y, event.ydata)
            self.point.set_data(x, y)
            self.line.set_data(x, y)

            if len(x) >= 6:

                if x[-1] == x[::5][-1]:
                    print("add")
                    addx = x[-1] - x[-2]
                    addy = y[-1] - y[-2]
                    x = np.append(x, x[-1] + addx)
                    y = np.append(y, y[-1] + addy)
                    self.point.set_data(x, y)
                    self.line.set_data(x, y)
                k = len(x[5::5])
                Cur = []

                for d in range(0, 5 * k, 5):
                    ra = np.array([x[d], y[d]])
                    rb = np.array([x[d + 1], y[d + 1]])
                    rc = np.array([x[d + 2], y[d + 2]])
                    rd = np.array([x[d + 3], y[d + 3]])
                    re = np.array([x[d + 4], y[d + 4]])
                    rf = np.array([x[d + 5], y[d + 5]])
                    # print(int(d/2))
                    Cur.append(biz5(ra, rb, rc, rd, re, rf))

                xc = np.array([])
                yc = np.array([])
                for d in range(k):
                    c = Cur[d]

                    xc = np.append(xc, c[::2])
                    yc = np.append(yc, c[1::2])
                self.curva.set_data(xc, yc)

            print(self.point.get_data())
        if event.button == 3:
            """Check whether mouse is over us; if so, store some data."""

            if event.inaxes != self.point.axes:
                return
            contains, attrd = self.point.contains(event)
            if not contains:
                return
            self.press = self.point.get_data(), (event.xdata, event.ydata)
        global pxc, pyc, px, py
        pxc = self.curva.get_xdata()
        pyc = self.curva.get_ydata()
        px = self.point.get_xdata()
        py = self.point.get_ydata()

    def onpick(self, event):

        ind = event.ind
        print('onpick points:', ind)

        self.i = ind[0]

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        # print(self.get_point())

        if self.press is None or event.inaxes != self.point.axes:
            return

        (x0, y0), (xpress, ypress) = self.press

        if (x0[self.i] in x0[4::5]) and (y0[self.i] in y0[4::5]) and (len(x0) >= 6):
            x0[self.i + 2] = 2 * x0[self.i + 1] - event.xdata
            y0[self.i + 2] = 2 * y0[self.i + 1] - event.ydata
        if (x0[self.i] in x0[6::5]) and (y0[self.i] in y0[6::5]):
            x0[self.i - 2] = 2 * x0[self.i - 1] - event.xdata
            y0[self.i - 2] = 2 * y0[self.i - 1] - event.ydata
        if (x0[self.i] in x0[5::5]) and (y0[self.i] in y0[5::5]):
            x0[self.i - 1] += event.xdata - x0[self.i]
            y0[self.i - 1] += event.ydata - y0[self.i]
            x0[self.i + 1] += event.xdata - x0[self.i]
            y0[self.i + 1] += event.ydata - y0[self.i]
        x0[self.i] = event.xdata
        y0[self.i] = event.ydata
        self.point.set_data(x0, y0)
        self.line.set_data(x0, y0)

        if len(x0) >= 6:
            k = len(x0[5::5])
            Cur = []

            for d in range(0, 5 * k, 5):
                ra = np.array([x0[d], y0[d]])
                rb = np.array([x0[d + 1], y0[d + 1]])
                rc = np.array([x0[d + 2], y0[d + 2]])
                rd = np.array([x0[d + 3], y0[d + 3]])
                re = np.array([x0[d + 4], y0[d + 4]])
                rf = np.array([x0[d + 5], y0[d + 5]])

                Cur.append(biz5(ra, rb, rc, rd, re, rf))

            x = np.array([])
            y = np.array([])
            for d in range(k):
                c = Cur[d]

                x = np.append(x, c[::2])
                y = np.append(y, c[1::2])

        self.curva.set_data(x, y)
        self.line.figure.canvas.draw()
        self.point.figure.canvas.draw()
        self.curva.figure.canvas.draw()

    def on_release(self, event):
        self.press = None

        self.point.figure.canvas.draw()
        self.line.figure.canvas.draw()
        self.curva.figure.canvas.draw()

        f = open('Cur.txt', 'w')

        for i in self.curva.get_data():

            np.savetxt(f,np.array([i]))
        f.close()

        f = open('Point.txt', 'w')

        for i in self.point.get_data():
            # print('i', len(i))
            np.savetxt(f, np.array([i]))
        f.close()


    def disconnect(self):
        """Disconnect all callbacks."""

        self.point.figure.canvas.mpl_disconnect(self.cid)
        self.point.figure.canvas.mpl_disconnect(self.cidpik)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

    def get_point(self):
        return self.point.get_data()

if __name__ == "__main__":

    fig, ax = plt.subplots()

    f = open("Point.txt", "r")
    px = np.array(f.readline().split()).astype(float)
    py = np.array(f.readline().split()).astype(float)
    f.close()

    f = open("Cur.txt", "r")
    pxc = np.array(f.readline().split()).astype(float)
    pyc = np.array(f.readline().split()).astype(float)
    f.close()

    curvaP = ax.plot([], [], linewidth=5, color="tab:orange")
    lineP = ax.plot([], [], linewidth=1, color="tab:blue")
    pointP = ax.plot([] ,[], 'ro', picker=True, pickradius=5)
    pointbuilder = PointBuilder(pointP[0], lineP[0], curvaP[0])
    pointbuilder.connect()

    plt.show()
