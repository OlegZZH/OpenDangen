import numpy as np
import pyrr.matrix44
from glfw import swap_buffers
from pyrr import Matrix44

from math import factorial as fc
import moderngl
from base import CameraWindow
from Lab1 import PointBuilder
from read import *
from OpenGL.GL import *
from moderngl_window import geometry
from random import *

from OpenGL.GLU import *

size = 100


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
        self.points_s = geometry.sphere(radius=0.5, sectors=32, rings=16)

        self.prog = self.load_program(
            vertex_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab2\resources\programs\vertex_shader.glsl",
            fragment_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab2\resources\programs\fragment_shader.glsl")

        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]
        self.L_M = self.prog["lookat"]
        self.T_M = self.prog["trans"]
        self.switcher = self.prog["switcher"]
        self.point=np.array([])
        self.curu=np.array([])
        self.index_curv=np.array([])
        self.add_surf()

        self.add_patch()

        self.vbo = self.ctx.buffer(grid(5, 15).astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))
        self.vbo_points = self.ctx.buffer(self.point.astype('f4'))

        self.vbo_curu = self.ctx.buffer(self.curu.astype('f4'))
        a = np.arange(8*4).reshape(8, 4)
        index = np.array([])
        for i in a:
            index = np.append(index, [i[0], *np.repeat(i[1:-1], 2), i[-1]])

        for j in range(0, len(a[0])):
            index = np.append(index, [a[:, j][0], *np.repeat(a[:, j][1:-1], 2), a[:, j][-1]])

        self.ibo_line = self.ctx.buffer(index.astype('i4'))
        self.ibo_curv = self.ctx.buffer(self.index_curv.astype('i4'))

        self.vao_grid = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_points = self.ctx.vertex_array(self.prog, self.vbo_points, 'in_vert', index_buffer=self.ibo_line)
        self.vao_curu = self.ctx.vertex_array(self.prog, [(self.vbo_curu, '3f', 'in_vert')], index_buffer=self.ibo_curv)

        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 4.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 1.0),  # up
        )
        self.translation = pyrr.matrix44.create_from_translation([0, 0, 0])

    def add_surf(self):
        h, w = 2, 2
        points = np.array([])
        for x in range(-h, h):
            for y in range(-w, w):
                points = np.append(points, [x, y, np.random.random() * 2])  # np.random.random()
        self.plot_surf(points)

    def add_patch(self):
        h, w = 2, 2
        new_points = np.array([])
        for x in range(-h, h):
            for y in range(-5, -1):
                new_points = np.append(new_points, [x, y, np.random.random() * 2])  # np.random.random()

        p = new_points.reshape(4, 4, 3)
        # print(self.point.reshape(4, 4, 3)[:, 0, :])
        p[:, -1, :] = self.point.reshape(4, 4, 3)[:, 0, :]
        # print(self.point.reshape(4, 4, 3)[:, 1, :])
        # print(self.point.reshape(4, 4, 3)[:, 0, :] - self.point.reshape(4, 4, 3)[:, 1, :])

        p[:, -2, :] = 2 * self.point.reshape(4, 4, 3)[:, 0, :] - self.point.reshape(4, 4, 3)[:, 1, :]

        # p[:, -2, :]=
        self.plot_surf(p)
        self.point= np.append(self.point, p)

    def plot_surf(self,points):
        curv = []
        curu = []
        for i in points.reshape(4, 4, 3):
            a = i[0]
            b = i[1]
            c = i[2]
            d = i[3]
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
        self.point = np.append(points,self.point )
        self.curu = np.append(curu,self.curu)
        # print(curu.reshape(size, size, 3))
        m=int(self.curu.size/(3*size**2))
        print("curv= ",self.curu.reshape(m*100,100,3))
        a = np.arange(m*size ** 2).reshape(m*size, size)
        print(len(a[0]))
        index = np.array([])
        for i in a:
            index = np.append(index, [i[0], *np.repeat(i[1:-1], 2), i[-1]])

        for j in a.T:
            index = np.append(index, [j[0], *np.repeat(j[1:-1], 2), j[-1]])


        self.index_curv= index



    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        glPointSize(10)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.L_M.write(self.lookat.astype('f4'))
        self.T_M.write(self.translation.astype('f4'))

        self.switcher.value = 0
        self.vao_grid.render(moderngl.LINES)
        self.switcher.value = 1
        self.vao_axis.render(moderngl.LINES)
        self.vao_points.render(moderngl.POINTS)
        self.vao_points.render(moderngl.LINES)

        self.switcher.value = 2
        # self.vao_curv.render(moderngl.LINES)
        # self.vao_curu.render(moderngl.POINTS)
        self.vao_curu.render(moderngl.LINES)


if __name__ == '__main__':
    SimpleGrid.run()
