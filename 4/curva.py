import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

t = np.linspace(0, 1, 100)


def biz(ra, rb, rc):
    r = np.array(
        [ra * ((1 - u) * (1 - u)) + (2 * rb * u * (1 - u)) + rc * (u ** 2) for u in t])

    return r


def biz3(ra, rb, rc, rd):
    r = np.array(
        [ra * ((1 - u) ** 3) + 3 * rb * u * ((1 - u) ** 2) + 3 * rc * (1 - u) * u ** 2 + rd * u ** 3 for u in t])

    return r


def ellipse(a, b, c):
    wa = 1
    f = 0.4
    wb = f / (1 - f)
    wc = 1

    r = np.array(
        [(a * wa * (1 - u) ** 2 + 2 * b * wb * u * (1 - u) + c * wc * u ** 2) / (
                wa * (1 - u) ** 2 + 2 * wb * u * (1 - u) + wc * u * u) for u in t])

    return r


def curva_3(ra, rb, rc,rd):
    wa = 1
    wb = 1
    wc = 1
    wd = 1
    r = np.array(
        [(ra * wa * (1 - u) ** 3 + 3 * rb * wb * u * (1 - u) ** 2 + 3 * rc * wc * u * u * (
                1 - u) +
          rd * wd * u ** 3) / (
                 wa * (1 - u) ** 3 + 3 * wb * u * (1 - u) ** 2 + 3 * wc * u * u * (1 - u) + wd * u ** 3)
         for u in t])

    return r


def cal_curva(x, y):
    c = np.empty((1, 2, 20))
    p=3
    if len(x) >= p+1:
        k = len(x[p::p])
        Cur = []

        for d in range(0, p * k, p):
            ra = np.array([x[d], y[d]])
            rb = np.array([x[d + 1], y[d + 1]])
            rc = np.array([x[d + 2], y[d + 2]])
            rd = np.array([x[d + 3], y[d + 3]])

            Cur.append(curva_3(ra, rb, rc,rd))

        c = np.array([[Cur[d][:, 0], Cur[d][:, 1]] for d in range(k)])
    return c


class PointBuilder:
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
            c = cal_curva(x, y)

            self.curva.set_data(c[:, 0], c[:, 1])
        if event.button == 3:
            """Check whether mouse is over us; if so, store some data."""

            if event.inaxes != self.point.axes:
                return
            contains, attrd = self.point.contains(event)
            if not contains:
                return
            self.press = self.point.get_data(), (event.xdata, event.ydata)

    def onpick(self, event):
        # contains, attrd = self.point.contains(event)
        # if not contains:
        #     return
        ind = event.ind
        print('onpick points:', ind)

        self.i = ind[0]

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        # print("on_motion")

        if self.press is None or event.inaxes != self.point.axes:
            return

        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        x0[self.i] = xpress + dx
        y0[self.i] = ypress + dy
        self.point.set_data(x0, y0)
        self.line.set_data(x0, y0)

        c = cal_curva(x0, y0)

        self.curva.set_data(c[:, 0], c[:, 1])
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
            np.savetxt(f, np.array(i).reshape(1, -1))
        f.close()

    def disconnect(self):
        """Disconnect all callbacks."""

        self.point.figure.canvas.mpl_disconnect(self.cid)
        self.point.figure.canvas.mpl_disconnect(self.cidpik)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    line = ax.plot([], linewidth=1, color="tab:blue")  # empty line
    # point, = ax.plot([], [], "ro", picker=True)
    point = ax.plot([], 'ro', picker=True, pickradius=5)
    curva = ax.plot([], linewidth=2, color="tab:orange")

    pointbuilder = PointBuilder(*point, *line, *curva)
    pointbuilder.connect()
    plt.gca().set_aspect("equal")
    plt.show()
