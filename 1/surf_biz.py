import multiprocessing
from pathlib import Path
import imgui
import moderngl
import numpy as np
import pyrr.matrix44
from OpenGL.GL import *
from PIL import ImageColor
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from pyrr import Matrix44
from scipy.spatial import Delaunay
from base import CameraWindow

"""
key:C -- camera(OFF/ON)
camera   |     point    |   axis
---------------------------------
key:W    |     key:I    |   z
key:S    |     key:K    |   z
key:A    |     key:J    |   y
key:D    |     key:L    |   y
key:Q    |     key:U    |   x
key:E    |     key:O    |   x
"""

face_color = np.array(ImageColor.getcolor("#4f596c", "RGB")) / 255
curva_color = np.array(ImageColor.getcolor("#00ff00", "RGB")) / 255
grid_color = np.array(ImageColor.getcolor("#eeeeee", "RGB")) / 255
point_color = np.array(ImageColor.getcolor("#FFB30F", "RGB")) / 255
axis_color = np.array(ImageColor.getcolor("#FF0000", "RGB")) / 255
pick_color = np.array(ImageColor.getcolor("#59C3C3", "RGB")) / 255



def terrain(size):
    vertices = np.dstack(np.mgrid[0:size, 0:size][::-1]) / size
    temp = np.dstack([np.arange(0, size * size - size), np.arange(size, size * size)])
    index = np.pad(temp.reshape(size - 1, 2 * size), [[0, 0], [0, 1]], 'constant', constant_values=-1)
    return vertices, index


def axis():
    buffer = np.array([[0, 10, 0], [0, -10, 0], [-10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 0, -10]])
    return buffer


class Surface:
    def __init__(self):
        self.point = np.array([])
        self.index = np.array([])
        self.size = 20
        self.t = np.linspace(0, 1, self.size)

    def add_surf(self, s_points=None):
        self.h, self.w = 7, 7

        self.point = np.array([[x, y, 0] for x in np.arange(0, self.h, 1) for y in np.arange(0, self.w, 1)])
        curv = self.plot_surf(self.point)
        self.tri = Delaunay(curv.reshape(-1, 3)[:, :2]).simplices

        if s_points != None:
            self.point = s_points['patch']
            self.point[:, :2] -= 2
            curv = self.plot_surf(self.point)

        self.update_index()
        return curv

    def plot_surf(self, points):
        p = 3

        kh = len(points.reshape(self.h, self.w, -1)[0][p::p])

        kw = len(points.reshape(self.h, self.w, -1)[:, 0][p::p])

        row_Cur_2 = np.array(
            [self.second_curve([i[j], i[j + 1], i[j + 2], i[j + 3]]) for i in points.reshape(self.h, self.w, -1) for j in
             range(0, kh * p, p)]).reshape(-1, 3)

        c = np.array(row_Cur_2).reshape(self.w, -1, 3)
        pool_of_points = np.array([[c[j][i], c[j + 1][i], c[j + 2][i], c[j + 3][i]] for i in range(self.size * kh) for j in
                                   range(0, kw * p, p)])
        Cur = np.array(pool.map(self.second_curve, pool_of_points))

        return Cur

    def second_curve(self, points):
        r = np.array(
            [points[0] * ((1 - u) ** 3) + 3 * points[1] * u * ((1 - u) ** 2) + 3 * points[2] * (1 - u) * u ** 2 + points[3] * u ** 3 for u in self.t])

        return r

    def update_index(self):
        index = np.array(
            [[(point_color[0] * 255 - i) / 255, point_color[1], point_color[2]] for i in range(len(self.point))])
        self.point = np.hstack((self.point, index))

    def plot_index(self):
        a = np.arange(self.h * self.w).reshape(self.h, self.w)
        self.index = [[i[0], *np.repeat(i[1:-1], 2), i[-1]] for i in a]
        self.index = np.append(self.index, [[j[0], *np.repeat(j[1:-1], 2), j[-1]] for j in a.T])

    def triangulation(self, curv):
        curv = curv.reshape(-1, 3)[self.tri]
        return curv


class SimpleGrid(CameraWindow):
    title = "Lab1"
    gl_version = (3, 3)
    resource_dir = (Path(__file__) / '../../Lab1/resources').resolve()

    fullscreen = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.wnd.ctx.error
        self.imgl = ModernglWindowRenderer(self.wnd)
        self.wnd.mouse_exclusivity = True
        self.camera.projection.update(near=0.01, far=100.0)
        self.camera.velocity = 5.0
        self.camera.mouse_sensitivity = 1
        self.shifts = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.size_range = (3, 50)

        self.sur = Surface()

        self.wW, self.hW = self.window_size
        self.select_color = np.array([])
        self.select_index = np.array([])

        self.prog = self.load_program(
            vertex_shader=r"programs\vertex_shader.glsl",
            fragment_shader=r"programs\fragment_shader.glsl")

        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]
        self.L_M = self.prog["lookat"]
        self.T_M = self.prog["trans"]
        self.switcher = self.prog["switcher"]

        curv = self.sur.add_surf(self.load_patch())  # self.load_patch()
        self.sur.plot_index()

        curv = self.sur.triangulation(curv)
        vertices, index = terrain(100)
        vertices[:, :, :] -= 0.5
        vertices *= 150

        self.vbo = self.ctx.buffer(vertices.astype('f4'))
        self.ibo = self.ctx.buffer(index.astype('i4'))
        vao_content = [
            (self.vbo, '2f', 'in_vert'),
        ]
        self.vbo_axis = self.ctx.buffer(axis().astype('f4') * 10)
        self.vbo_points = self.ctx.buffer(self.sur.point.astype('f4'))
        self.vbo_curu = self.ctx.buffer(curv.astype('f4'), dynamic=True)

        self.ibo_line = self.ctx.buffer(self.sur.index.astype('i4'))

        self.vao_grid = self.ctx.vertex_array(self.prog, vao_content, self.ibo)
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_points = self.ctx.vertex_array(self.prog, [(self.vbo_points, "3f 3f", 'in_vert', "point_color")],
                                                self.ibo_line)
        self.vao_curu = self.ctx.vertex_array(self.prog, [(self.vbo_curu, '3f', 'in_vert')])

        self.lookat = Matrix44.look_at(
            (-10, -10, 1.0),  # eye
            (0, 0, 1.0),  # target
            (.0, .0, 1.0),  # up
            dtype='f4')
        self.trans = pyrr.matrix44.create_from_y_rotation(0)
        io = imgui.get_io()
        self.new_font = io.fonts.add_font_from_file_ttf(
            "StripedSansBlack.ttf", 35
        )
        self.imgl.refresh_font_texture()

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))
        self.imgl.mouse_press_event(x, y, button)
        data = glReadPixels(x, self.hW - y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        print(data[0], data[1], data[2])
        if self.select_color.shape[0] != 0:
            find = np.where(np.around(self.sur.point[:, 3] * 255) == np.around(pick_color[0] * 255))
            self.sur.point[find[0], 3:] = self.select_color
            self.select_color = np.array([])

        if [data[0], data[1], data[2]] != [255, 255, 255] and data[1] == point_color[1] * 255 and data[2] == \
                point_color[2] * 255:
            find = np.where(np.around(self.sur.point[:, 3] * 255) == data[0])
            self.select_color = self.sur.point[find[0]][:, 3:]
            self.sur.point[find[0], 3:] = pick_color

        self.vbo_points.write(self.sur.point.astype("f4"))

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgl.mouse_drag_event(x, y, dx, dy)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgl.mouse_release_event(x, y, button)

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        def find():
            return np.where(np.around(self.sur.point[:, 3] * 255) == np.around(pick_color[0] * 255))

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled

            if key == keys.SPACE:
                self.timer.toggle_pause()
            if self.select_color.shape[0] != 0:
                if key == keys.I:
                    self.sur.point[find()[0], 2] -= 0.25
                if key == keys.K:
                    self.sur.point[find()[0], 2] += 0.25
                if key == keys.J:
                    self.sur.point[find()[0], 1] -= 0.25
                if key == keys.L:
                    self.sur.point[find()[0], 1] += 0.25
                if key == keys.O:
                    self.sur.point[find()[0], 0] -= 0.25
                if key == keys.U:
                    self.sur.point[find()[0], 0] += 0.25
                curv = self.sur.triangulation(self.sur.plot_surf(self.sur.point[:, :3]))

                self.vbo_curu.write(curv.astype('f4'))
                self.vbo_points.write(self.sur.point.astype("f4"))

                # self.save_patch()

    def save_patch(self):
        np.savez("surface1", patch=self.sur.point[:, :3])

    def load_patch(self):
        return np.load("surface1.npz")

    def render(self, time, frame_time):
        self.ctx.clear(*face_color)
        self.ctx.enable_only(moderngl.DEPTH_TEST)  # moderngl.CULL_FACE |
        glPointSize(15)
        # self.P_M.write(pyrr.matrix44.create_orthogonal_projection(-16, 16,-9, 9,-2000, 2000,dtype='f4'))
        self.P_M.write(Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0, dtype='f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.L_M.write(self.lookat.astype('f4'))
        self.T_M.write(pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0.0, 0])).astype('f4'))
        self.ctx.wireframe = True
        self.switcher.value = pyrr.Vector3(grid_color, dtype='f4')
        self.vao_grid.render(moderngl.TRIANGLE_STRIP)
        self.switcher.value = pyrr.Vector3(axis_color, dtype='f4')
        self.vao_axis.render(moderngl.LINES)
        self.T_M.write(self.trans)
        self.switcher.value = pyrr.Vector3(curva_color, dtype='f4')
        self.vao_curu.render(moderngl.TRIANGLES, vertices=int(1e6))
        self.switcher.value = pyrr.Vector3([1, 2, 3], dtype='f4')
        self.vao_points.render(moderngl.POINTS)
        self.switcher.value = pyrr.Vector3(axis_color, dtype='f4')
        self.vao_points.render(moderngl.LINES)
        self.ctx.wireframe = False
        self.render_ui()

    def render_ui(self):

        imgui.new_frame()

        # imgui.show_test_window()
        imgui.set_next_window_bg_alpha(0.7)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *grid_color)
        imgui.push_style_color(imgui.COLOR_TEXT, *axis_color)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *grid_color)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *grid_color)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *grid_color)
        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10.0)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10.0)
        imgui.push_font(self.new_font)

        imgui.begin("Transformations")
        if imgui.tree_node("Translation", imgui.TREE_NODE_DEFAULT_OPEN):
            _, self.shifts[0] = imgui.input_float("X", self.shifts[0], 0.25)
            _, self.shifts[1] = imgui.input_float("Y", self.shifts[1], 0.25)
            _, self.shifts[2] = imgui.input_float("Z", self.shifts[2], 0.25)
            imgui.tree_pop()
        if imgui.tree_node("Rotation", imgui.TREE_NODE_DEFAULT_OPEN):
            _, self.angles[0] = imgui.slider_angle("X", self.angles[0], -180, 180)
            _, self.angles[1] = imgui.slider_angle("Y", self.angles[1], -180, 180)
            _, self.angles[2] = imgui.slider_angle("Z", self.angles[2], -180, 180)
            imgui.tree_pop()
        changed, self.sur.size = imgui.slider_int("Size ", self.sur.size, *self.size_range, "%.2f")

        imgui.end()
        imgui.pop_font()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_var()
        imgui.pop_style_var()

        rot = pyrr.matrix44.create_from_z_rotation(self.angles[2],
                                                   dtype='f4') @ pyrr.matrix44.create_from_y_rotation(
            self.angles[1], dtype='f4') @ pyrr.matrix44.create_from_x_rotation(self.angles[0],
                                                                               dtype='f4')

        self.trans = pyrr.matrix44.create_from_translation(
            pyrr.Vector3([self.shifts[0], self.shifts[1], self.shifts[2]], dtype='f4')) @ rot

        imgui.render()
        self.imgl.render(imgui.get_draw_data())
        if changed:
            self.sur.t = np.linspace(0, 1, self.sur.size)
            curv = self.sur.add_surf(self.load_patch())
            curv = self.sur.triangulation(curv)
            self.vbo_curu.orphan(len(curv.flatten()) * 4)
            self.vbo_curu.write(curv.astype('f4').tobytes())


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    SimpleGrid.run()
