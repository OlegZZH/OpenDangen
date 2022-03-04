import numpy as np
import pyrr.matrix44
from glfw import swap_buffers
from pyrr import Matrix44

import moderngl
from base import CameraWindow
from Lab1 import PointBuilder
from read import *
from OpenGL.GL import *
from moderngl_window import geometry

from OpenGL.GLU import *


def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


def line(px, py):
    line_buffer = []
    point_T=[]
    red=255
    for x, y in zip(px, py):
        line_buffer.append([y, 0, x,0,(red/255)/50,0])
        point_T.append(pyrr.matrix44.create_from_translation([y,0,x]))
        red-=1
    line_buffer = np.array(line_buffer)
    point_T=np.array(point_T)
    print(line_buffer * 50)
    print(point_T * 50)
    return line_buffer * 50,point_T*50




def curva(pxc, pyc):
    curva_buffer = []
    for x, y in zip(pxc, pyc):
        curva_buffer.append([y, 0, x])
    curva_buffer = np.array(curva_buffer)
    # print(curva_buffer)
    return curva_buffer * 50


def sur(pxc, pyc):
    surface_W = []
    for i, j in zip(pxc, pyc):
        for p in np.linspace(0, 2 * np.pi):
            surface_W.append([np.cos(p) * j * 50, np.sin(p) * j * 50, i * 50])
    surface_W = np.array(surface_W)
    # print(surface_W)
    return surface_W


def axis():
    buffer = np.array([[0, 10, 0], [0, -10, 0], [-10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 0, -10]])
    # print(buffer)
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
        line_buffer, point_T = line(px, py)

        self.prog = self.load_program(
            vertex_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab1\resources\programs\vertex_shader.glsl",
            fragment_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab1\resources\programs\fragment_shader.glsl")

        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]

        self.L_M = self.prog["lookat"]
        self.R_M = self.prog["rot_Z"]
        self.switcher = self.prog["switcher"]

        self.vbo = self.ctx.buffer(grid(5, 15).astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))
        self.vbo_line = self.ctx.buffer(line_buffer.astype("f4"))
        self.vbo_curva = self.ctx.buffer(curva(pxc, pyc).astype("f4"))
        self.sur = self.ctx.buffer(sur(pxc, pyc).astype("f4"))

        self.vao_grid = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_line = self.ctx.vertex_array(self.prog, [(self.vbo_line, "3f 3f",'in_vert',"point_color")],)
        self.vao_curva = self.ctx.vertex_array(self.prog, self.vbo_curva, 'in_vert')
        self.vao_sur = self.ctx.vertex_array(self.prog, self.sur, 'in_vert')


        self.h = 100
        self.surface_H = np.array([pyrr.matrix44.create_from_z_rotation(0)])
        for i in np.linspace(0, 2 * np.pi, self.h):
            self.surface_H = np.append(self.surface_H, np.array([pyrr.matrix44.create_from_z_rotation(i)]), axis=0)

        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 15.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 1.0),  # up
        )

    def set_buf(self):

        self.vbo_line.clear()

        self.vbo_line.write(line([0, 0, 1, 2], [0, 2, 3, 4]).astype("f4"))
        self.vbo_curva.clear()
        self.vbo_curva.write(curva([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 5, 5]).astype("f4"))
        self.sur.clear()
        self.sur.write(line([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 5, 5]).astype("f4"))

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        glPointSize(10)
        rot_Z = pyrr.matrix44.create_from_z_rotation(time)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)

        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        # print(self.camera.matrix.astype('f4'))
        self.L_M.write(self.lookat.astype('f4'))

        self.switcher.value = 0
        self.vao_grid.render(moderngl.LINES)
        self.switcher.value = 2
        self.vao_sur.render(moderngl.LINE_STRIP)

        self.switcher.value = 1
        self.vao_axis.render(moderngl.LINES)

        self.R_M.write(rot_Z.astype('f4'))
        # self.vao_line.render(moderngl.LINE_STRIP)
        # self.vao_curva.render(moderngl.LINE_STRIP)
        # self.switcher.value = 3
        # self.vao_line.render(moderngl.POINTS)
        #
        #
        # self.switcher.value = 2
        # for i in self.surface_H:
        #     self.R_M.write(i.astype('f4'))
        #     self.vao_curva.render(moderngl.LINE_STRIP)
        #     # self.vao_line.render(moderngl.LINE_STRIP)
        #     # self.vao_line.render(moderngl.POINTS)



if __name__ == '__main__':
    SimpleGrid.run()