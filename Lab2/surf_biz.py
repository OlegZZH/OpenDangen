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
from random import *

from OpenGL.GLU import *

def surf(h,w):
    line = np.array([])
    row =[]
    for x  in range(-h,h):
        for y in range(-w,w):
            line=np.append(line,[x,y,0])#np.random.random()
        row.append(line)
    return line

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


        self.vbo = self.ctx.buffer(grid(5, 15).astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))
        self.vbo_points=self.ctx.buffer(surf(2,2).astype('f4'))
        a=np.arange(0,16).reshape(4,4)
        index=np.array([])
        indey=np.array([])

        for i in a :
            index=np.append(index,[i[0],*np.repeat(i[1:-1],2),i[-1]])

        for j in range(0,len(a[0])):
            print(a[:,j])
            index = np.append(index, [a[:,j][0], *np.repeat(a[:,j][1:-1], 2), a[:,j][-1]])

        print(index)
        self.ibo_line=self.ctx.buffer(index.astype('i4'))

        self.vao_grid = self.ctx.vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_points = self.ctx.vertex_array(self.prog, self.vbo_points, 'in_vert',index_buffer=self.ibo_line )

        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 15.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 1.0),  # up
        )
        self.translation=pyrr.matrix44.create_from_translation([0,0,0])



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





if __name__ == '__main__':
    SimpleGrid.run()