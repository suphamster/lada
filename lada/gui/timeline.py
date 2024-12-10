import pathlib

from gi.repository import Gtk, GObject, Gdk, Graphene, Gsk, Adw

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(filename=here / 'timeline.ui')
class Timeline(Gtk.Widget):
    __gtype_name__ = 'Timeline'

    @GObject.Property(type=Adw.Application)
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        self._application = value
        self.style_manager = value.get_property("style-manager")

    @GObject.Property()
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self.update_duration(value)
        self.update_playhead_position(0)

    @GObject.Property()
    def playhead_position(self):
        return self._playhead_position

    @playhead_position.setter
    def playhead_position(self, value):
        self.update_playhead_position(value)

    @GObject.Signal(name="seek_requested", arg_types=(GObject.TYPE_INT,))
    def seek_requested_signal(self, position: int):
        pass

    @GObject.Signal(name="cursor_position_changed", arg_types=(GObject.TYPE_INT,))
    def cursor_position(self, position: int | None):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._application = None
        self._playhead_position = 0
        self.cursor_position_x: int | None = None
        self._duration = 0
        self.video_file = None
        self.set_hexpand(True)
        self.set_css_name('timeline')

        self.gesture_drag = Gtk.GestureDrag.new()
        self.drag_start = 0
        def on_drag_begin(gesture_drag, x, y):
            gesture_drag.set_state( Gtk.EventSequenceState.CLAIMED)
            self.on_drag_begin(x)
        self.gesture_drag.connect("drag-begin", on_drag_begin)
        self.gesture_drag.connect("drag-end", lambda _, offset_x, offset_y: self.on_drag_end(offset_x))
        self.add_controller(self.gesture_drag)

        event_controller_motion = Gtk.EventControllerMotion.new()
        event_controller_motion.connect("leave", lambda _: self.update_cursor_position(None))
        event_controller_motion.connect("motion", lambda _, x, y: self.update_cursor_position(x))
        self.add_controller(event_controller_motion)

        self.style_manager = None

    def update_duration(self, value):
        self._duration = value
        self._playhead_position = 0
        self.queue_draw()

    def update_playhead_position(self, value):
        self._playhead_position = value
        self.queue_draw()

    def on_drag_end(self, offset_x):
        x = self.drag_start + offset_x
        x = max(0, x)
        allocation = self.get_allocation()
        width = allocation.width
        new_position = int((x / width) * self._duration)
        self.update_playhead_position(new_position)
        self.emit('seek_requested', new_position)

    def on_drag_begin(self, x):
        self.drag_start = x

    def update_cursor_position(self, x):
        self.cursor_position_x = x
        if x:
            allocation = self.get_allocation()
            width = allocation.width
            cursor_position = int((x / width) * self._duration)
        else:
            cursor_position = -1
        self.queue_draw()
        self.emit('cursor_position_changed', cursor_position)

    def do_snapshot(self, s):
        """
        AFAIK, it's currently only possible to get the accent color programmatically. Other colors need apparently be to be
        hardcoded and pray that it somewhat matches the documented Adwaita colors (unless another theme is used)
        https://gnome.pages.gitlab.gnome.org/libadwaita/doc/main/css-variables.html
        """
        allocation = self.get_allocation()
        width = allocation.width
        height = allocation.height

        playhead_position_x = min(int((self._playhead_position / self._duration) * width), width - 1) if self._duration > 0 else 0

        if self.style_manager:
            playhead_color = self.style_manager.get_accent_color()
            uses_dark_scheme = bool(self.style_manager.get_dark())
        else:
            playhead_color = Adw.AccentColor.BLUE
            uses_dark_scheme = False
        playhead_color = playhead_color.to_rgba(playhead_color)

        timeline_color = Gdk.RGBA()
        cursor_color = Gdk.RGBA()
        if uses_dark_scheme:
            timeline_color.parse("#ffffff1a")
            cursor_color.parse("#ffffffff")
        else:
            timeline_color.parse("#0000001a")
            cursor_color.parse("#000000ff")

        cursor_width = 2
        playhead_width = 4
        before_width = playhead_position_x - playhead_width // 2
        after_width = width - playhead_width - before_width

        rect_height = height

        rounded_corner = Graphene.Size()
        rounded_corner.init(10, 10)
        regular_corner = Graphene.Size()
        regular_corner.init(0, 0)

        before = Graphene.Rect().init(0, 0, before_width, rect_height)
        before_rounded = Gsk.RoundedRect()
        before_rounded.init(before, rounded_corner, regular_corner, regular_corner, rounded_corner)
        s.push_rounded_clip(before_rounded)
        s.append_color(timeline_color, before)
        s.pop()

        playhead_position = Graphene.Rect().init(before_width, 0, playhead_width, rect_height)
        s.append_color(playhead_color, playhead_position)

        after = Graphene.Rect().init(before_width + playhead_width, 0, after_width,rect_height)
        after_rounded = Gsk.RoundedRect()
        after_rounded.init(after, regular_corner, rounded_corner, rounded_corner, regular_corner)
        s.push_rounded_clip(after_rounded)
        s.append_color(timeline_color, after)
        s.pop()

        if self.cursor_position_x:
            x = self.cursor_position_x - (cursor_width // 2)
            width = width - x if x + cursor_width > width else cursor_width
            cursor_position = Graphene.Rect().init(x, 0, width, rect_height)
            s.append_color(cursor_color, cursor_position)