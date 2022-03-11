import numpy as np
import pyrr.matrix44
from glfw import swap_buffers
from pyrr import Matrix44

from math import factorial as fc
import moderngl
from base import CameraWindow

from OpenGL.GL import *
from moderngl_window import geometry
from random import *

from OpenGL.GLU import *

size = 50
color = 255


def biz(*args):
    t = np.linspace(0, 1, size)
    r = []
    for u in t:
        tm = (1 - u)
        r = np.append(r, (args[0] * tm ** 3) + (3 * u * args[1] * tm ** 2) + (3 * args[2] * tm * u ** 2) + args[
            3] * u ** 3)
    return r


def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


def axis():
    buffer = np.array([[0, 10, 0], [0, -10, 0], [-10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 0, -10]])
    return buffer


class SimpleGrid(CameraWindow):
    title = "Simple Grid"
    gl_version = (3, 3)

    def __init__(self, **args):
        super().__init__(**args)
        self.wnd.mouse_exclusivity = True
        self.camera.projection.update(near=0.01, far=100.0)
        self.camera.velocity = 5.0
        self.camera.mouse_sensitivity = 0.3
        self.select_color = np.array([])
        self.select_index = np.array([])

        self.prog = self.load_program(
            vertex_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab2\resources\programs\vertex_shader.glsl",
            fragment_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab2\resources\programs\fragment_shader.glsl")

        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]
        self.L_M = self.prog["lookat"]
        self.T_M = self.prog["trans"]
        self.switcher = self.prog["switcher"]
        self.point = np.array([])
        self.curu = np.array([])
        self.index_curv = np.array([])
        self.index = np.array([])
        self.load_patch()
        self.add_surf(self.load_data)
        self.add_patch(self.load_data)
        self.update_index()
        self.vbo = self.ctx.buffer(grid(5, 15).astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))
        self.vbo_points = self.ctx.buffer(self.point.astype('f4'))
        self.vbo_curu = self.ctx.buffer(self.curu.astype('f4'))

        self.ibo_line = self.ctx.buffer(self.index.astype('i4'))
        self.ibo_curv = self.ctx.buffer(self.index_curv.astype('i4'))

        self.vao_grid = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_points = self.ctx.vertex_array(self.prog, [(self.vbo_points, "3f 3f", 'in_vert', "point_color")],
                                                index_buffer=self.ibo_line)
        self.vao_curu = self.ctx.vertex_array(self.prog, [(self.vbo_curu, '3f', 'in_vert')], index_buffer=self.ibo_curv)

        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 4.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 1.0),  # up
        )
        self.translation = pyrr.matrix44.create_from_translation([0, 0, 0])

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))
        w, h = self.window_size

        data = glReadPixels(x, h - y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        print(data[0], data[1], data[2])
        if self.select_color.shape[0] != 0:
            find = np.where((self.point[:, :, 3] * 255).astype(int) == 0)
            self.point[find[0], find[1], 3:] = self.select_color
            self.select_color = np.array([])

        if [data[0], data[1], data[2]] != [255, 255, 255] and data[1] == 0 and data[2] == 0:
            find = np.where((self.point[:, :, 3] * 255).astype(int) == data[0])
            self.select_color = self.point[find[0], find[1], :][:, 3:]
            self.point[find[0], find[1], 3:] = 0

        self.vbo_points.write(self.point.astype("f4"))

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        def find():
            return np.where((self.point[:, :, 3] * 255).astype(int) == 0)

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()
            if not self.camera_enabled and self.select_color.shape[0] != 0:
                if key == keys.I:
                    self.point[find()[0], find()[1], 2] -= 0.1
                if key == keys.K:
                    self.point[find()[0], find()[1], 2] += 0.1
                if key == keys.J:
                    self.point[find()[0], find()[1], 1] -= 0.1
                if key == keys.L:
                    self.point[find()[0], find()[1], 1] += 0.1
                if key == keys.O:
                    self.point[find()[0], find()[1], 0] -= 0.1
                if key == keys.U:
                    self.point[find()[0], find()[1], 0] += 0.1

                self.curu = np.array([])
                self.curu = np.append(self.plot_surf(self.point[:, :, :3][1::2]), self.curu)
                self.curu = np.append(self.plot_surf(self.point[:, :, :3][::2]), self.curu)

                self.vbo_curu.write(self.curu.astype('f4'))
                self.vbo_points.write(self.point.astype("f4"))

                self.save_patch()
    def save_patch(self):
        np.savez("surface1",patch1=self.point[:, :, :3][1::2],patch2= self.point[:, :, :3][::2])

    def load_patch(self):
        self.load_data=np.load("surface1.npz")




    def update_index(self):
        global color
        index = np.array([])
        for i in range(self.point.shape[0] * self.point.shape[1]):
            index = np.append(index, [color / 255, 0, 0])
            color -= 1
        self.point = np.append(self.point, index.reshape(self.point.shape), axis=2)

    def add_surf(self,s_points=None):
        h, w = 2, 2
        points = np.array([])
        if s_points==None:
            for x in range(-h, h):
                for y in range(-w, w):
                    points = np.append(points, [x, y, 0])
        else :
            points=s_points['patch1']
        self.curu = np.append(self.plot_surf(points), self.curu)
        self.point = np.append(points, self.point)

    def add_patch(self,s_points=None):
        h, w = 2, 2
        new_points = np.array([])
        if s_points==None:
            for x in range(-h, h):
                for y in range(-5, -1):
                    new_points = np.append(new_points, [x, y, 0])
        else:
            new_points=s_points["patch2"]
        p = new_points.reshape(4, 4, 3)
        # p[:, -1, :] = self.point.reshape(4, 4, 3)[:, 0, :]
        # p[:, -2, :] = 2 * self.point.reshape(4, 4, 3)[:, 0, :] - self.point.reshape(4, 4, 3)[:, 1, :]


        self.curu = np.append(self.plot_surf(p.flatten()), self.curu)
        self.plot_index()
        a = np.arange(8 * 4).reshape(8, 4)
        for i in a:
            self.index = np.append(self.index, [i[0], *np.repeat(i[1:-1], 2), i[-1]])

        for j in a.reshape(4, 8).T:
            self.index = np.append(self.index, [j[0], *np.repeat(j[1:-1], 2), j[-1]])
        temp = np.reshape(self.point, [int(self.point.size / 12), 4, 3])
        temp = np.insert(temp, [0, 1, 2, 3], [p[0], p[1], p[2], p[3]], axis=0)
        self.point = temp

    def plot_surf(self, points):
        curv = []
        curu = []
        for i in points.reshape(4, 4, 3):
            a = i[0][0:3]
            b = i[1][0:3]
            c = i[2][0:3]
            d = i[3][0:3]
            curv.append(biz(a, b, c, d))

        curv = np.array(curv)
        cf = curv.reshape(4, size, 3)

        for i in range(len(cf[0])):
            a = cf[0][i]
            b = cf[1][i]
            c = cf[2][i]
            d = cf[3][i]

            curu.append(biz(a, b, c, d))
        curu = np.array(curu)
        return curu

    def plot_index(self):
        m = int(self.curu.size / (3 * size ** 2))
        a = np.arange(m * size ** 2).reshape(m * size, size)
        index = np.array([])
        for i in a:
            index = np.append(index, [i[0], *np.repeat(i[1:-1], 2), i[-1]])

        for j in a.T:
            index = np.append(index, [j[0], *np.repeat(j[1:-1], 2), j[-1]])

        self.index_curv = index

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        glPointSize(15)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.L_M.write(self.lookat.astype('f4'))
        self.T_M.write(self.translation.astype('f4'))

        self.switcher.value = 0
        self.vao_grid.render(moderngl.LINES)
        self.switcher.value = 1
        self.vao_axis.render(moderngl.LINES)

        self.switcher.value = 2
        self.vao_curu.render(moderngl.LINES)
        self.switcher.value = 3
        self.vao_points.render(moderngl.POINTS)
        self.switcher.value = 1
        self.vao_points.render(moderngl.LINES)


if __name__ == '__main__':
    SimpleGrid.run()
