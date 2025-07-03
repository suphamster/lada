import sys
import threading
import time
from gi.repository import Gtk, GObject


class FullscreenMouseActivityController(GObject.Object):
    def __init__(self, fullscreen_widget):
        GObject.Object.__init__(self)
        self.fullscreen_widget = fullscreen_widget
        self.fullscreen_motion_controller = None
        self.activity_timer = None
        self.motion_started_time = None
        self.last_motion_x: float = 0.
        self.last_motion_y: float = 0.
        self._fullscreen_activity = False
        self.idle_time_seconds = 1.5
        self.fullscreen_activated = None
        self.initial_grace_period = 0.2

    @GObject.Property()
    def fullscreen_activity(self):
        return self._fullscreen_activity

    @fullscreen_activity.setter
    def fullscreen_activity(self, value):
        self._fullscreen_activity = value

    def on_fullscreened(self, fullscreened: bool):
        self.fullscreen_activated  = time.time()
        if fullscreened:
            self.fullscreen_motion_controller = Gtk.EventControllerMotion.new()
            self.fullscreen_motion_controller.connect("motion", self._on_motion)
            self.fullscreen_widget.add_controller(self.fullscreen_motion_controller)
            self.last_motion_y = sys.maxsize
            self.last_motion_y = sys.maxsize
            self._start_timer()
        else:
            self.fullscreen_widget.remove_controller(self.fullscreen_motion_controller)
            if self.activity_timer:
                self.activity_timer.cancel()
                self.activity_timer = None

    def on_activity_timer_run(self, *args):
        self.motion_started_time = None
        self.activity_timer = None
        self.fullscreen_activity = False

    def _on_motion(self, obj, x, y):
        if not self._is_considerable_mouse_motion(x, y):
            return
        if self.fullscreen_activated and ((time.time() - self.fullscreen_activated) < self.initial_grace_period):
            return
        self.motion_started_time = time.time()

        if not self._fullscreen_activity:
            self.fullscreen_activity = True

        self._start_timer()
        self.last_motion_y = y
        self.last_motion_x = x

    def _start_timer(self):
        if self.activity_timer:
            self.activity_timer.cancel()
        self.activity_timer = threading.Timer(self.idle_time_seconds, self.on_activity_timer_run)
        self.activity_timer.start()

    def _is_considerable_mouse_motion(self, x, y, min_distance_px=3.):
        return abs(x - self.last_motion_x) >= min_distance_px or abs(y - self.last_motion_y) >= min_distance_px