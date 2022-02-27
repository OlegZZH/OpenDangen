import numpy as np
import pyrr.matrix44
from glfw import swap_buffers
from pyrr import Matrix44

import moderngl
from base import CameraWindow

from read import *
from OpenGL.GL import *
from moderngl_window import geometry

from OpenGL.GLU import *


def biz_sur(*args):
    t = np.linspace(0, 1, 40)
    r = []
    for u in t:
        r.append(args[0] * (1 - u) ** 5 + 5 * u * args[1] * (1 - u) ** 4 + (10 * args[2] * u ** 2) * (
                1 - u) ** 3 + (10 * args[3] * u ** 3) * (1 - u) ** 2 + (5 * args[4] * u ** 4) * (1 - u) + args[
                     5] * u ** 5)
    return r


def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


def axis():
    buffer = np.array([[0, 10, 0], [0, -10, 0], [-10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 0, -10]])
    # print(buffer)
    return buffer


def line(px, py,pz,*color):
    line_x = []
    red = color[1]
    i = 0

    c=np.zeros(3)
    c[color[0]]=color[1]
    for x, y ,z in zip(px, py,pz):
        line_x.append([x,y, z, *c])
        c[color[0]] =c[color[0]]- 1
    line_x = np.array(line_x)
    return line_x


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

        lineU = line(50*pxU[:-1:], 50*pyU[:-1:],np.zeros_like(pxU[:-1:]),0,255)
        lineV = line(50*pxV[:-1:],np.zeros_like(pxU[:-1:]), 50*pyV[:-1:],1,255)

        curvaU = []
        curvaV = []
        # lineV = np.flip(lineV[:, :3], axis=1)
        # print(lineV)
        # print(lineV)
        for i in range(len(lineV[::6])):
            a = np.array(lineV[i + 0])
            b = np.array(lineV[i + 1])
            c = np.array(lineV[i + 2])
            d = np.array(lineV[i + 3])
            e = np.array(lineV[i + 4])
            f = np.array(lineV[i + 5])
            curvaV.append(biz_sur(a[:3], b[:3], c[:3], d[:3], e[:3], f[:3]))
        curvaV = np.array(*curvaV)

        for i in range(len(lineU[::6])):
            a = np.array(lineU[i + 0])
            b = np.array(lineU[i + 1])
            c = np.array(lineU[i + 2])
            d = np.array(lineU[i + 3])
            e = np.array(lineU[i + 4])
            f = np.array(lineU[i + 5])
            curvaU.append(biz_sur(a[:3], b[:3], c[:3], d[:3], e[:3], f[:3]))
        curvaU = np.array(*curvaU)



        self.surface_U = np.array([pyrr.matrix44.create_from_translation(curvaU[0])])
        for i in curvaU[1::]:
            self.surface_U = np.append(self.surface_U, np.array([pyrr.matrix44.create_from_translation(i)]), axis=0)

        self.surface_V = np.array([pyrr.matrix44.create_from_translation(curvaV[0])])
        for i in curvaV[1::]:
            self.surface_V = np.append(self.surface_V, np.array([pyrr.matrix44.create_from_translation(i)]), axis=0)



        print(len(self.surface_U))
        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]
        self.L_M = self.prog["lookat"]
        self.T_M = self.prog["trans"]
        self.switcher = self.prog["switcher"]

        self.vbo = self.ctx.buffer(grid(5, 15).astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))
        self.vbo_lineU = self.ctx.buffer(lineU.astype("f4"))
        self.vbo_curU = self.ctx.buffer(curvaU.astype("f4"))
        self.vbo_lineV = self.ctx.buffer(lineV.astype("f4"))
        self.vbo_curV = self.ctx.buffer(curvaV.astype("f4"))

        self.vao_grid = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_lineU = self.ctx.vertex_array(self.prog, [(self.vbo_lineU, "3f 3f", 'in_vert', "point_color")])
        self.vao_curU = self.ctx.vertex_array(self.prog, self.vbo_curU, 'in_vert')
        self.vao_lineV = self.ctx.vertex_array(self.prog, [[self.vbo_lineV, "3f 3f", 'in_vert', "point_color"]])
        self.vao_curV = self.ctx.vertex_array(self.prog, self.vbo_curV, 'in_vert')

        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 15.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 1.0),  # up
        )

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        glPointSize(10)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)

        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.L_M.write(self.lookat.astype('f4'))
        # self.T_M.write(self.surface_U[0].astype('f4'))
        self.switcher.value = 0
        self.vao_grid.render(moderngl.LINES)

        self.switcher.value = 1
        self.vao_axis.render(moderngl.LINES)

        self.switcher.value = 3
        self.T_M.write(self.surface_V[0].astype('f4'))
        self.vao_lineU.render(moderngl.POINTS)
        self.vao_lineU.render(moderngl.LINE_STRIP)
        self.T_M.write(self.surface_U[0].astype('f4'))
        self.vao_lineV.render(moderngl.POINTS)
        self.vao_lineV.render(moderngl.LINE_STRIP)

        for i in self.surface_U:
            self.T_M.write(i.astype('f4'))
            self.switcher.value = 2
            self.vao_curV.render(moderngl.LINE_STRIP)

        for i in self.surface_V:
            self.T_M.write(i.astype('f4'))
            self.switcher.value = 2
            self.vao_curU.render(moderngl.LINE_STRIP)


if __name__ == '__main__':
    SimpleGrid.run()
