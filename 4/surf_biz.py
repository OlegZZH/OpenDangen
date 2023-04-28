import multiprocessing
from pathlib import Path

import imgui
import moderngl
import numpy as np
import pyrr.matrix44
from OpenGL.GL import *
from PIL import ImageColor
from moderngl_window import geometry
from pyrr import Matrix44
from scipy.spatial import Delaunay

from base import CameraWindow
from read import pxc, pyc

"""
camera   |     point    |   axis
---------------------------------
key:W    |     key:I    |   z
key:S    |     key:K    |   z
key:A    |     key:J    |   y
key:D    |     key:L    |   y
key:Q    |     key:U    |   x
key:E    |     key:O    |   x
"""

wa = 1
f = 0.6
wb = f / (1 - f)
wc = 1

face_color = np.array(ImageColor.getcolor("#1F363D", "RGB")) / 255
curva_color = np.array(ImageColor.getcolor("#c2f2a0", "RGB")) / 255
grid_color = np.array(ImageColor.getcolor("#40798C", "RGB")) / 255
point_color = np.array(ImageColor.getcolor("#9EC1A3", "RGB")) / 255
axis_color = np.array(ImageColor.getcolor("#DAB6FC", "RGB")) / 255
pick_color = np.array(ImageColor.getcolor("#23a9dd", "RGB")) / 255

shininess = 16
light_position = np.array([0, 3, 3], dtype='f4')
light_ambient = np.array([0.2, 0.2, 0.2], dtype='f4')
light_diffuse = np.array([0.8, 0.8, 0.8], dtype='f4')
light_specular = np.array([1.0, 1.0, 1.0], dtype='f4')
constant = 1.0
linear = 0.027
quadratic = 0.0028


def triangle_normal(triangle):
    a, b, c = triangle
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    return normalize_vector(normal)


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    else:
        return vector / norm


def terrain(size):
    vertices = np.dstack(np.mgrid[0:size, 0:size][::-1]) / size
    temp = np.dstack([np.arange(0, size * size - size), np.arange(size, size * size)])
    index = np.pad(temp.reshape(size - 1, 2 * size), [[0, 0], [0, 1]], 'constant', constant_values=-1)
    return vertices, index


def axis():
    buffer = np.array([[0, 10, 0], [0, -10, 0], [-10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 0, -10]])
    return buffer


class Surface:
    size = 25
    t = np.linspace(0, 1, size)

    def __init__(self):
        self.y_range = None
        self.x_range = None
        self.yrange = None
        self.xrange = None
        self.tri = np.array([])
        self.point = np.array([])
        self.curves = np.array([])
        self.line_index = np.array([])
        self.polygons = np.array([])

        self.h, self.w = 7, 7
        self.t_map = []

    def add_point(self, s_points=None):
        if s_points is not None:
            self.point = s_points
            self.point[:, :2] -= 3
        else:
            self.point = np.array([[x, y, 0] for x in np.arange(0, self.h, 1) for y in np.arange(0, self.w, 1)])

    def calc_triangulation(self):
        self.tri = Delaunay(self.curves[:, :2]).simplices

    def plot_surf(self):
        p = 3

        kh = len(self.point.reshape(self.h, self.w, -1)[0][p::p])

        kw = len(self.point.reshape(self.h, self.w, -1)[:, 0][p::p])

        row_Cur_2 = np.array(
            [self.curva([i[j], i[j + 1], i[j + 2], i[j + 3]]) for i in self.point.reshape(self.h, self.w, -1) for j in
             range(0, kh * p, p)]).reshape(-1, 3)

        c = np.array(row_Cur_2).reshape(self.w, -1, 3)
        pool_of_points = np.array(
            [[c[j][i], c[j + 1][i], c[j + 2][i], c[j + 3][i]] for i in range(self.size * kh) for j in
             range(0, kw * p, p)])
        self.curves = np.array(pool.map(self.curva, pool_of_points)).reshape(-1, 3)

    def curva(self, points):
        r = np.array(
            [points[0] * ((1 - u) ** 3) + 3 * points[1] * u * ((1 - u) ** 2) + 3 * points[2] * (1 - u) * u ** 2 +
             points[3] * u ** 3 for u in self.t])

        return r

    def update_index(self):
        index = np.array(
            [[(point_color[0] * 255 - i) / 255, point_color[1], point_color[2]] for i in range(len(self.point))])
        self.point = np.hstack((self.point, index))

    def plot_index(self):
        a = np.arange(self.h * self.w).reshape(self.h, self.w)
        self.line_index = [[i[0], *np.repeat(i[1:-1], 2), i[-1]] for i in a]
        self.line_index = np.append(self.line_index, [[j[0], *np.repeat(j[1:-1], 2), j[-1]] for j in a.T])

    def point_to_polygons(self, curv):
        self.polygons = curv[self.tri]

    def normals(self):
        n = np.tile(np.array(pool.map(triangle_normal, self.polygons[:, :, :3])), 3).reshape(-1, 3)

        self.polygons = np.c_[self.polygons.reshape(-1, 5), n]

    def texture_mapping(self, xrange, yrange):
        self.xrange = xrange
        self.yrange = yrange
        self.x_max = np.max(self.curves[:, 0])
        self.x_min = np.min(self.curves[:, 0])
        self.y_min = np.min(self.curves[:, 1])
        self.x_range = np.max(self.curves[:, 0]) - np.min(self.curves[:, 0])
        self.y_range = np.max(self.curves[:, 1]) - np.min(self.curves[:, 1])

        self.t_map = np.array(pool.map(self.text_m, self.curves))

    def text_m(self, point):
        if np.isclose(point[0], self.xrange[1]):
            point[0] = self.xrange[1]
        elif np.isclose(point[0], self.xrange[0]):
            point[0] = self.xrange[0]
        elif np.isclose(point[1], self.yrange[1]):
            point[1] = self.yrange[1]
        elif np.isclose(point[1], self.yrange[0]):
            point[1] = self.yrange[0]
        if self.xrange[1] > point[0] > self.xrange[0] and self.yrange[1] > point[1] > self.yrange[0]:
            u = (self.x_max - point[0] - self.x_min) / self.x_range
            v = (point[1] - self.y_min) / self.y_range
        else:
            u, v = 0, 0
        return u, v


def load_patch():
    return np.load("surface1.npz")['patch']


class SimpleGrid(CameraWindow):
    title = "Simple Grid"
    gl_version = (3, 3)
    resource_dir = (Path(__file__) / '../../4/resources').resolve()

    fullscreen = True

    def __init__(self, **args):
        super().__init__(**args)
        self.curve_i = 1
        self.point_in_curve = 0
        self.changed_button = [False, False, False]
        self.radio_active = [True, False, False, False]
        self.wnd.mouse_exclusivity = True
        self.camera.projection.update(near=0.01, far=100.0)
        self.wnd.ctx.error
        self.camera.velocity = 5.0
        self.camera.mouse_sensitivity = 0.3
        self.select_color = np.array([])
        self.select_index = np.array([])

        self.shifts = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.size_range = (3, 50)
        self.xrange, self.yrange = [0, 6], [0, 2]
        self.patch = load_patch()

        self.sun = geometry.sphere(radius=0.05)
        self.prog = self.load_program(vertex_shader=r"programs\vertex_shader.glsl",
                                      fragment_shader=r"programs\fragment_shader.glsl")
        self.sun_prog = self.load_program('programs/cube_simple.glsl')
        self.sun_prog['color'].value = 1, 1, 1, 1
        self.sun_prog['m_model'].write(Matrix44.from_translation(np.array([0.0, 0.0, 0.0], dtype="f4")))
        self.P_M = self.prog["prog"]
        self.C_M = self.prog["cam"]
        self.L_M = self.prog["lookat"]
        self.T_M = self.prog["trans"]
        self.switcher = self.prog["switcher"]
        self.viewPos = self.prog["viewPos"]
        self.shininess = self.prog["material.shininess"]
        self.light_position = self.prog["light.position"]
        self.light_ambient = self.prog["light.ambient"]
        self.light_diffuse = self.prog["light.diffuse"]
        self.light_specular = self.prog["light.specular"]
        self.light_constant = self.prog["light.constant"]
        self.light_linear = self.prog["light.linear"]
        self.light_quadratic = self.prog["light.quadratic"]

        self.texture = self.load_texture_2d(
            r'texture/Metallic_surface_texture.jpg')

        self.empty_sur = Surface()
        self.empty_sur.add_point()
        self.empty_sur.plot_surf()
        self.empty_sur.calc_triangulation()
        self.empty_sur.texture_mapping(self.xrange, self.yrange)

        self.sur = Surface()
        self.sur.add_point(self.patch)  # self.patch
        self.sur.plot_surf()
        self.sur.tri = self.empty_sur.tri
        self.sur.t_map = self.empty_sur.t_map
        self.sur.point_to_polygons(np.c_[self.sur.curves, self.sur.t_map])
        self.sur.normals()
        self.sur.update_index()
        self.sur.plot_index()

        vertices, ind = terrain(30)
        vertices[:, :, :] -= 0.5
        vertices *= 100

        self.vbo = self.ctx.buffer(vertices.astype('f4'))
        self.ibo_g = self.ctx.buffer(ind.astype('i4'))

        self.vbo_points = self.ctx.buffer(self.sur.point.astype('f4'))
        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))
        self.vbo_curva = self.ctx.buffer(np.c_[pxc, pyc, np.zeros_like(pxc)].astype('f4'))

        self.vbo_polygon = self.ctx.buffer(self.sur.polygons.astype('f4'))

        self.ibo_line = self.ctx.buffer(self.sur.line_index.astype('i4'))

        self.vao_grid = self.ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_vert')], self.ibo_g)
        self.vao_points = self.ctx.vertex_array(self.prog, [(self.vbo_points, "3f 3f", 'in_vert', "point_color")],
                                                index_buffer=self.ibo_line)
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')
        self.vao_curva = self.ctx.vertex_array(self.prog, self.vbo_curva, 'in_vert')

        self.vao_polygon = self.ctx.vertex_array(self.prog,
                                                 [(self.vbo_polygon, '3f 2f 3f', 'in_vert', "text_cord", "normal")])

        self.lookat = Matrix44.look_at(
            (0.01, 0.0, 5.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 0.0, 5.0),  # up
        )
        self.trans = pyrr.matrix44.create_from_translation([0, 0, 0])
        self.L_M.write(self.lookat.astype('f4'))
        self.sun_prog['lookat'].write(self.lookat.astype('f4'))

        self.shininess.value = shininess
        self.light_position.write(light_position)
        self.light_ambient.write(light_ambient)
        self.light_diffuse.write(light_diffuse)
        self.light_specular.write(light_specular)
        self.light_constant.value = constant
        self.light_linear.value = linear
        self.light_quadratic.value = quadratic
        io = imgui.get_io()
        self.new_font = io.fonts.add_font_from_file_ttf(
            "Blazed.ttf", 30,
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

                self.sur.point, index = self.sur.point[:, :3], self.sur.point[:, 3:]
                self.sur.plot_surf()
                self.sur.point_to_polygons(np.c_[self.sur.curves, self.sur.t_map])
                self.sur.normals()
                self.sur.point = np.c_[self.sur.point, index]
                self.vbo_polygon.write(self.sur.polygons.astype('f4'))
                self.vbo_points.write(self.sur.point.astype("f4"))

                # self.save_patch()

    def save_patch(self):
        np.savez("surface1", patch=self.sur.point[:, :3])

    def render(self, time, frame_time):
        if self.point_in_curve < len(pxc) - self.curve_i:
            self.point_in_curve += self.curve_i
        else:
            self.point_in_curve = 0
        self.ctx.clear(*face_color)
        self.ctx.enable_only(moderngl.DEPTH_TEST)  # moderngl.CULL_FACE |
        glPointSize(15)
        self.texture.use()
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)

        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.T_M.write(pyrr.matrix44.create_from_translation([0, 0, 0]).astype('f4'))
        self.viewPos.write(self.camera.position.astype('f4'))

        self.sun_prog['m_proj'].write(proj.astype('f4'))
        self.sun_prog['m_camera'].write(self.camera.matrix.astype('f4'))
        self.sun_prog['m_model'].write(Matrix44.from_translation(np.array(self.light_position.value), dtype='f4'))
        self.sun.render(self.sun_prog)

        self.ctx.wireframe = True
        self.switcher.write(grid_color.astype('f4'))
        self.vao_grid.render(moderngl.TRIANGLE_STRIP)
        self.ctx.wireframe = False
        self.switcher.write(axis_color.astype('f4'))
        self.vao_axis.render(moderngl.LINES)
        self.vao_curva.render(moderngl.LINES)
        self.switcher.value = pyrr.Vector3([1, 1, 1], dtype='f4')

        self.T_M.write(self.trans)
        self.switcher.write(np.array([1, 2, 3]).astype('f4'))
        self.vao_points.render(moderngl.POINTS)
        self.switcher.write(axis_color.astype('f4'))
        self.vao_points.render(moderngl.LINES)

        self.switcher.write(np.array([6, 0, 1]).astype('f4'))
        self.vao_polygon.render(moderngl.TRIANGLES, vertices=int(2e5))
        self.render_ui()

    def render_ui(self):
        imgui.new_frame()

        # imgui.show_test_window()
        # imgui.set_next_window_bg_alpha(0.7)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *grid_color)
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.7)

        imgui.push_style_color(imgui.COLOR_TEXT, *axis_color)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *grid_color)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *grid_color)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *grid_color)
        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10.0)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10.0)
        imgui.push_font(self.new_font)

        imgui.begin("Euclidean transformations")
        if imgui.tree_node("Translation", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.begin_group()
            imgui.text("X")
            _, self.shifts[0] = imgui.v_slider_float("##1", 150, 300, self.shifts[0], -15, 15, "%.2f")
            imgui.end_group()
            imgui.same_line()
            imgui.begin_group()
            imgui.text("Y")
            _, self.shifts[1] = imgui.v_slider_float("##2", 150, 300, self.shifts[1], -15, 15, "%.2f")
            imgui.end_group()
            imgui.same_line()
            imgui.begin_group()
            imgui.text("Z")
            _, self.shifts[2] = imgui.v_slider_float("##3", 150, 300, self.shifts[2], -15, 15, "%.2f")
            imgui.end_group()
            imgui.tree_pop()
        if imgui.tree_node("Rotation", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.begin_group()
            imgui.text("X")
            _, self.angles[0] = imgui.v_slider_float("##1", 150, 300, self.angles[0], -180, 180, "%.2f")
            imgui.end_group()
            imgui.same_line()
            imgui.begin_group()
            imgui.text("Y")
            _, self.angles[1] = imgui.v_slider_float("##2", 150, 300, self.angles[1], -180, 180, "%.2f")
            imgui.end_group()
            imgui.same_line()
            imgui.begin_group()
            imgui.text("Z")
            _, self.angles[2] = imgui.v_slider_float("##3", 150, 300, self.angles[2], -180, 180, "%.2f")
            imgui.end_group()
            imgui.tree_pop()

        changed, Surface.size = imgui.slider_int("Size ", Surface.size, *self.size_range, "%.2f")
        imgui.end()

        imgui.begin("Texture")
        self.changed_button[0] = imgui.button("1", 65, 200)
        imgui.same_line()
        self.changed_button[1] = imgui.button("2", 65, 200)
        imgui.same_line()
        self.changed_button[2] = imgui.button("3", 65, 200)
        imgui.end()

        imgui.begin("Illumination")
        changed_d, d = imgui.slider_float3("Position ", *self.light_position.value, -10, 10, "%.2f")
        _, self.light_constant.value = imgui.slider_float("Constant", self.light_constant.value, 0, 1)
        _, self.light_linear.value = imgui.slider_float("Linear", self.light_linear.value, 0, 1)
        _, self.light_quadratic.value = imgui.slider_float("Quadratic", self.light_quadratic.value, 0, 1)
        imgui.end()

        imgui.begin("Animation")
        _, self.point_in_curve = imgui.slider_int("Curve translation", self.point_in_curve, 0, len(pxc) - 1)
        _, self.curve_i = imgui.slider_int("SpeedX", self.curve_i, 0, 4)
        imgui.end()

        imgui.pop_font()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_var()
        imgui.pop_style_var()
        imgui.pop_style_var()

        if any(self.changed_button):
            if self.changed_button[0]:
                self.xrange, self.yrange = [0, 6], [0, 2]
            if self.changed_button[1]:
                self.xrange, self.yrange = [0, 6], [2, 4]
            if self.changed_button[2]:
                self.xrange, self.yrange = [0, 6], [4, 6]

            self.empty_sur.texture_mapping(self.xrange, self.yrange)
            self.sur.t_map = self.empty_sur.t_map
            self.empty_sur.point_to_polygons(np.array(self.sur.t_map))
            self.sur.polygons[:, 3:5] = self.empty_sur.polygons.reshape(-1, 2)
            self.vbo_polygon.write(self.sur.polygons.astype('f4'))

        rot = pyrr.matrix44.create_from_z_rotation(np.deg2rad(self.angles[2]),
                                                   dtype='f4') @ pyrr.matrix44.create_from_y_rotation(
            np.deg2rad(self.angles[1]), dtype='f4') @ pyrr.matrix44.create_from_x_rotation(np.deg2rad(self.angles[0]),
                                                                                           dtype='f4')

        self.trans = pyrr.matrix44.create_from_translation(
            pyrr.Vector3([self.shifts[0], self.shifts[1], self.shifts[2]],
                         dtype='f4')) @ rot @ pyrr.matrix44.create_from_translation(
            [pxc[self.point_in_curve], pyc[self.point_in_curve], 0], dtype='f4')

        imgui.render()
        self.imgl.render(imgui.get_draw_data())
        if changed:
            Surface.t = np.linspace(0, 1, Surface.size)
            self.empty_sur.plot_surf()
            self.empty_sur.calc_triangulation()
            self.empty_sur.texture_mapping(self.xrange, self.yrange)

            self.sur.point, index = self.sur.point[:, :3], self.sur.point[:, 3:]
            self.sur.plot_surf()
            self.sur.point = np.c_[self.sur.point, index]
            self.sur.tri = self.empty_sur.tri
            self.sur.t_map = self.empty_sur.t_map
            self.sur.point_to_polygons(np.c_[self.sur.curves, self.sur.t_map])
            self.sur.normals()

            self.vbo_polygon.orphan(len(self.sur.polygons.flatten()) * 4)
            self.vbo_polygon.write(self.sur.polygons.astype('f4'))
        if changed_d:
            self.light_position.write(np.array(d).astype("f4"))


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    SimpleGrid.run()
