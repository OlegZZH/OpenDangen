import imgui
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from moderngl_window.scene.camera import KeyboardCamera, OrbitCamera
from moderngl_window.scene import Scene
from OpenGL.GL import *

from OpenGL.GLU import *

class CameraWindow(mglw.WindowConfig):
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True
        imgui.create_context()
        self.imgl = ModernglWindowRenderer(self.wnd)
        self.wnd.ctx.error
        self.wW, self.hW = self.window_size


    def mouse_position_event(self, x: int, y: int, dx, dy):

        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
        self.wW, self.hW = width, height
        self.imgl.resize(width, height)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgl.mouse_drag_event(x, y, dx, dy)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgl.mouse_release_event(x, y, button)


class OrbitCameraWindow(mglw.WindowConfig):
    """Base class with built in 3D orbit camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = OrbitCamera(aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if self.camera_enabled:
            self.camera.zoom_state(y_offset)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
