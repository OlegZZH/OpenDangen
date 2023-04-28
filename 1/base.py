import moderngl_window as mglw
from moderngl_window.scene.camera import KeyboardCamera, OrbitCamera


class CameraWindow(mglw.WindowConfig):
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True


    def mouse_position_event(self, x: int, y: int, dx, dy):

        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
        print("Window was resized. buffer size is {} x {}".format(width, height))
        self.wW, self.hW=width, height
        self.imgl.resize(width, height)



class OrbitCameraWindow(mglw.WindowConfig):
    """Base class with built in 3D orbit camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = OrbitCamera(aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True



    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if self.camera_enabled:
            self.camera.zoom_state(y_offset)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
        print("Window was resized. buffer size is {} x {}".format(width, height))
        self.wW, self.hW=width, height
