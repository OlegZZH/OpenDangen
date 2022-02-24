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


        self.prog = self.load_program(
            vertex_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab2\resources\programs\vertex_shader.glsl",
            fragment_shader=r"C:\Users\Oleg\Dropbox\lab\OpenDangen\Lab2\resources\programs\fragment_shader.glsl")

        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]

        self.L_M = self.prog["lookat"]

        self.switcher = self.prog["switcher"]

        self.vbo = self.ctx.buffer(grid(5, 15).astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))


        self.vao_grid = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')



        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 15.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 1.0),  # up
        )


    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        glPointSize(10)
        rot_Z = pyrr.matrix44.create_from_z_rotation(time)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)

        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.L_M.write(self.lookat.astype('f4'))

        self.switcher.value = 0
        self.vao_grid.render(moderngl.LINES)

        self.switcher.value = 1
        self.vao_axis.render(moderngl.LINES)









if __name__ == '__main__':
    SimpleGrid.run()