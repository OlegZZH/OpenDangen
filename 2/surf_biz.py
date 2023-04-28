import multiprocessing
from pathlib import Path

import imgui
import moderngl
import numpy as np
import pyrr.matrix44
from OpenGL.GL import *
from PIL import ImageColor
from moderngl_window import geometry
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from pyrr import Matrix44
from scipy.spatial import Delaunay

from base import CameraWindow

"""
camera   |     points    |   axis
---------------------------------
key:W    |     key:I    |   z
key:S    |     key:K    |   z
key:A    |     key:J    |   y
key:D    |     key:L    |   y
key:Q    |     key:U    |   x
key:E    |     key:O    |   x
"""

face_color = np.array(ImageColor.getcolor("#1C0118", "RGB")) / 255
curva_color = np.array(ImageColor.getcolor("#6BD425", "RGB")) / 255
grid_color = np.array(ImageColor.getcolor("#618B25", "RGB")) / 255
point_color = np.array(ImageColor.getcolor("#BFBDC1", "RGB")) / 255
axis_color = np.array(ImageColor.getcolor("#FF4D80", "RGB")) / 255
pick_color = np.array(ImageColor.getcolor("#45425A", "RGB")) / 255
ambient = np.array([0.4125, 0.435, 0.425]).astype('f4')
diffuse = np.array([0.5038, 0.5048, 0.528]).astype('f4')
specular = np.array([0.777, 0.622, 0.6014]).astype('f4')
shininess = 512
light_position = np.array([0, 0, -3], dtype='f4')
light_ambient = np.array([0.2, 0.2, 0.2], dtype='f4')
light_diffuse = np.array([0.8, 0.8, 0.8], dtype='f4')
light_specular = np.array([1.0, 1.0, 1.0], dtype='f4')

wa = 1
f = 0.6
wb = f / (1 - f)
wc = 1


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
    size2 = 20
    t2 = np.linspace(0, 1, size2)
    size4 = size2 * 2
    t4 = np.linspace(0, 1, size4)

    def __init__(self):
        self.points = np.array([])
        self.curves = np.array([])
        self.line_index = np.array([])
        self.polygons = np.array([])
        self.tri_index = np.array([])

        self.h, self.w = 7, 7


    def add_point(self, s_points=None):
        if s_points is not None:
            self.points = s_points
            # self.points[:, :2] += 2
        else:
            self.points = np.array([[x, y, 0] for x in np.arange(0, self.h, 1) for y in np.arange(0, self.w, 1)])

    def calc_triangulation(self):
        self.tri_index = Delaunay(self.curves[:, :2]).simplices

    def plot_surf(self):
        p2 = 2
        p4 = 3
        kh2 = len(self.points.reshape(self.h, self.w, -1)[0][p2::p2])
        kw4 = len(self.points.reshape(self.h, self.w, -1)[:, 0][p4::p4])

        row_Cur_2 = np.array(
            [self.biz2([i[j], i[j + 1], i[j + 2]]) for i in self.points.reshape(self.h, self.w, -1) for j in
             range(0, kh2 * p2, p2)]).reshape(-1, 3)

        c = row_Cur_2.reshape(self.w, -1, 3)
        pool_of_points = np.array(
            [[c[j][i], c[j + 1][i], c[j + 2][i], c[j + 3][i]] for i in range(self.size2 * kh2) for j in
             range(0, kw4 * p4, p4)])
        self.curves = np.array(pool.map(self.biz3, pool_of_points)).reshape(-1, 3)

    def biz2(self, points):
        r = np.array(
            [points[0] * ((1 - u) * (1 - u)) + (2 * points[1] * u * (1 - u)) + points[2] * (u ** 2) for u in self.t2])

        return r

    def biz3(self, points):
        a, b, c, d = points
        r = np.array(
            [a * ((1 - u) ** 3) + 3 * b * u * ((1 - u) ** 2) + 3 * c * (1 - u) * u ** 2 + d * u ** 3 for u in self.t4])

        return r

    def update_index(self):
        index = np.array(
            [[(point_color[0] * 255 - i) / 255, point_color[1], point_color[2]] for i in range(len(self.points))])
        self.points = np.hstack((self.points, index))

    def plot_index(self):
        a = np.arange(self.h * self.w).reshape(self.h, self.w)
        self.line_index = [[i[0], *np.repeat(i[1:-1], 2), i[-1]] for i in a]
        self.line_index = np.append(self.line_index, [[j[0], *np.repeat(j[1:-1], 2), j[-1]] for j in a.T])

    def point_to_polygons(self, curv):
        self.polygons = curv[self.tri_index]

    def normals(self):
        n = np.tile(np.array(pool.map(triangle_normal, self.polygons[:, :, :3])), 3).reshape(-1, 3)

        self.polygons = np.c_[self.polygons.reshape(-1, 3), n]


def load_patch():
    return np.load("surface1.npz")['patch']


class SimpleGrid(CameraWindow):
    title = "Simple Grid"
    gl_version = (3, 3)
    resource_dir = (Path(__file__) / '../../2/resources').resolve()

    fullscreen = True

    def __init__(self, **args):
        super().__init__(**args)
        self.wnd.mouse_exclusivity = True
        imgui.create_context()
        self.wnd.ctx.error
        self.imgl = ModernglWindowRenderer(self.wnd)
        self.camera.projection.update(near=0.01, far=100.0)
        self.camera.velocity = 5.0
        self.camera.mouse_sensitivity = 0.3
        self.select_color = np.array([])
        self.select_index = np.array([])
        self.wW, self.hW = self.window_size
        self.shifts = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.size_range = (3, 50)
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
        self.objectColor = self.prog["objectColor"]
        self.viewPos = self.prog["viewPos"]
        self.ambient = self.prog["material.ambient"]
        self.diffuse = self.prog["material.diffuse"]
        self.specular = self.prog["material.specular"]
        self.shininess = self.prog["material.shininess"]
        self.lightPos = self.prog["light.position"]
        self.light_ambient = self.prog["light.ambient"]
        self.light_diffuse = self.prog["light.diffuse"]
        self.light_specular = self.prog["light.specular"]

        self.empty_sur = Surface()
        self.empty_sur.add_point()
        self.empty_sur.plot_surf()
        self.empty_sur.calc_triangulation()

        self.sur = Surface()
        self.sur.add_point(self.patch)  # self.patch
        self.sur.plot_surf()
        self.sur.tri_index = self.empty_sur.tri_index

        self.sur.point_to_polygons(self.sur.curves)
        self.sur.normals()
        self.sur.update_index()
        self.sur.plot_index()
        vertices, ind = terrain(30)
        vertices[:, :, :] -= 0.5
        vertices *= 100

        self.vbo = self.ctx.buffer(vertices.astype('f4'))
        self.ibo_g = self.ctx.buffer(ind.astype('i4'))
        vao_content = [
            (self.vbo, '2f', 'in_vert'),
        ]

        self.vbo_axis = self.ctx.buffer(axis().astype('f4'))

        self.vbo_points = self.ctx.buffer(self.sur.points.astype('f4'))

        self.vbo_polygon = self.ctx.buffer(self.sur.polygons.astype('f4'))

        self.ibo_line = self.ctx.buffer(self.sur.line_index.astype('i4'))

        self.vao_grid = self.ctx.vertex_array(self.prog, vao_content, self.ibo_g)
        self.vao_axis = self.ctx.vertex_array(self.prog, self.vbo_axis, 'in_vert')

        self.vao_points = self.ctx.vertex_array(self.prog, [(self.vbo_points, "3f 3f", 'in_vert', "point_color")],
                                                index_buffer=self.ibo_line)
        self.vao_polygon = self.ctx.vertex_array(self.prog, [(self.vbo_polygon, "3f 3f", 'in_vert', "normal")])

        self.lookat = Matrix44.look_at(
            (6, 0, 1.0),  # eye
            (0, 0, 1.0),  # target
            (.0, .0, 1.0),  # up
        )
        self.trans = pyrr.matrix44.create_from_translation([0, 0, 0])
        self.L_M.write(self.lookat.astype('f4'))
        self.sun_prog['lookat'].write(self.lookat.astype('f4'))
        self.objectColor.write(curva_color.astype('f4'))
        self.ambient.write(ambient)
        self.diffuse.write(diffuse)
        self.specular.write(specular)
        self.shininess.value = shininess
        self.lightPos.write(light_position)
        self.light_ambient.write(light_ambient)
        self.light_diffuse.write(light_diffuse)
        self.light_specular.write(light_specular)

        io = imgui.get_io()
        self.new_font = io.fonts.add_font_from_file_ttf(
            "True Lies.ttf", 30,
        )
        self.imgl.refresh_font_texture()

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))
        self.imgl.mouse_press_event(x, y, button)
        data = glReadPixels(x, self.hW - y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        print(data[0], data[1], data[2])
        if self.select_color.shape[0] != 0:
            find = np.where(np.around(self.sur.points[:, 3] * 255) == np.around(pick_color[0] * 255))
            self.sur.points[find[0], 3:] = self.select_color
            self.select_color = np.array([])

        if [data[0], data[1], data[2]] != [255, 255, 255] and data[1] == point_color[1] * 255 and data[2] == \
                point_color[2] * 255:
            find = np.where(np.around(self.sur.points[:, 3] * 255) == data[0])
            self.select_color = self.sur.points[find[0]][:, 3:]
            self.sur.points[find[0], 3:] = pick_color

        self.vbo_points.write(self.sur.points.astype("f4"))

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgl.mouse_drag_event(x, y, dx, dy)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgl.mouse_release_event(x, y, button)

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        def find():
            return np.where(np.around(self.sur.points[:, 3] * 255) == np.around(pick_color[0] * 255))

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
                    self.sur.points[find()[0], 2] -= 0.25
                if key == keys.K:
                    self.sur.points[find()[0], 2] += 0.25
                if key == keys.J:
                    self.sur.points[find()[0], 1] -= 0.25
                if key == keys.L:
                    self.sur.points[find()[0], 1] += 0.25
                if key == keys.O:
                    self.sur.points[find()[0], 0] -= 0.25
                if key == keys.U:
                    self.sur.points[find()[0], 0] += 0.25
                self.sur.points, index = self.sur.points[:, :3], self.sur.points[:, 3:]
                self.sur.plot_surf()
                self.sur.point_to_polygons(self.sur.curves)
                self.sur.normals()
                self.sur.points = np.c_[self.sur.points, index]
                self.vbo_polygon.write(self.sur.polygons.astype('f4'))
                self.vbo_points.write(self.sur.points.astype("f4"))

            # self.save_patch()

    def save_patch(self):
        np.savez("surface1", patch=self.sur.points[:, :3])

    def render(self, time, frame_time):
        self.ctx.clear(*face_color)
        self.ctx.enable_only(moderngl.DEPTH_TEST)  # moderngl.CULL_FACE |
        glPointSize(15)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)

        self.P_M.write(proj.astype('f4'))
        self.C_M.write(self.camera.matrix.astype('f4'))
        self.T_M.write(pyrr.matrix44.create_from_translation([0, 0, 0]).astype('f4'))
        self.viewPos.write(self.camera.position.astype('f4'))

        x = np.cos(2 * time) * 5 + 3
        y = 3
        z = 5

        self.lightPos.write(np.array([x, y, z]).astype('f4'))
        self.sun_prog['m_proj'].write(proj.astype('f4'))
        self.sun_prog['m_camera'].write(self.camera.matrix.astype('f4'))
        self.sun_prog['m_model'].write(Matrix44.from_translation([x, y, z], dtype='f4'))
        self.sun.render(self.sun_prog)

        self.ctx.wireframe = True
        self.switcher.write(pyrr.Vector3(grid_color, dtype='f4'))
        self.vao_grid.render(moderngl.TRIANGLE_STRIP)
        self.ctx.wireframe = False
        self.switcher.write(pyrr.Vector3(axis_color, dtype='f4'))
        self.vao_axis.render(moderngl.LINES)
        self.switcher.write(pyrr.Vector3([1, 1, 1], dtype='f4'))

        self.T_M.write(self.trans)
        self.switcher.write(pyrr.Vector3([6, 0, 1], dtype='f4'))
        self.vao_polygon.render(moderngl.TRIANGLES, vertices=int(1e6))
        self.switcher.write(pyrr.Vector3([1, 2, 3], dtype='f4'))
        self.vao_points.render(moderngl.POINTS)
        self.switcher.write(pyrr.Vector3(axis_color, dtype='f4'))
        self.vao_points.render(moderngl.LINES)
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

        imgui.begin("Euclidean transformations")
        _, self.shifts = imgui.slider_float3("Translation", *self.shifts, -20, 20,
                                             "%.2f")
        _, self.angles = imgui.slider_float3("Rotation", *self.angles, -180, 180,
                                             "%.2f")
        changed, Surface.size2 = imgui.slider_int("Size ", Surface.size2, *self.size_range, "%.2f")

        if imgui.tree_node("Light", imgui.TREE_NODE_DEFAULT_OPEN):
            _, self.light_ambient.value = imgui.slider_float3("Ambient", *self.light_ambient.value, 0, 1)
            _, self.light_diffuse.value = imgui.slider_float3("Diffuse", *self.light_diffuse.value, 0, 1)
            _, self.light_specular.value = imgui.slider_float3("Specular", *self.light_specular.value, 0, 1)
            _, self.shininess.value = imgui.slider_float("Shininess", self.shininess.value, 2, 512)
            imgui.tree_pop()
        if imgui.tree_node("Material", imgui.TREE_NODE_DEFAULT_OPEN):
            _, self.objectColor.value = imgui.slider_float3("Object Color", *self.objectColor.value, 0, 1)
            _, self.ambient.value = imgui.slider_float3("Ambient", *self.ambient.value, 0, 1)
            _, self.diffuse.value = imgui.slider_float3("Diffuse", *self.diffuse.value, 0, 1)
            _, self.specular.value = imgui.slider_float3("Specular", *self.specular.value, 0, 1)
            imgui.tree_pop()

        imgui.end()
        imgui.pop_font()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.pop_style_var()
        imgui.pop_style_var()

        rot = pyrr.matrix44.create_from_z_rotation(np.deg2rad(self.angles[2]),
                                                   dtype='f4') @ pyrr.matrix44.create_from_y_rotation(
            np.deg2rad(self.angles[1]), dtype='f4') @ pyrr.matrix44.create_from_x_rotation(np.deg2rad(self.angles[0]),
                                                                                           dtype='f4')

        self.trans = pyrr.matrix44.create_from_translation(
            pyrr.Vector3([self.shifts[0], self.shifts[1], self.shifts[2]], dtype='f4')) @ rot

        imgui.render()
        self.imgl.render(imgui.get_draw_data())
        if changed:
            Surface.t2 = np.linspace(0, 1, Surface.size2)
            self.empty_sur.plot_surf()
            self.empty_sur.calc_triangulation()

            self.sur.points, index = self.sur.points[:, :3], self.sur.points[:, 3:]
            self.sur.plot_surf()
            self.sur.points = np.c_[self.sur.points, index]
            self.sur.tri_index = self.empty_sur.tri_index
            self.sur.point_to_polygons(self.sur.curves)
            self.sur.normals()

            self.vbo_polygon.orphan(len(self.sur.polygons.flatten()) * 4)
            self.vbo_polygon.write(self.sur.polygons.astype('f4'))


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    SimpleGrid.run()
