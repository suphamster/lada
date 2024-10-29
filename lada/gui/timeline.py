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
        self.update_position(0)

    @GObject.Property()
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self.update_position(value)

    @GObject.Signal(name="seek_requested", arg_types=(GObject.TYPE_INT,))
    def seek_requested_signal(self, position: int):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._application = None
        self._position = 0
        self._duration = 0
        self.video_file = None
        self.set_hexpand(True)
        self.set_css_name('timeline')

        self.gesture_drag = Gtk.GestureDrag.new()
        self.drag_start = 0
        def on_drag_begin(gesture_drag, x, y):
            gesture_drag.set_state( Gtk.EventSequenceState.CLAIMED)
            self.on_drag_begin(x, y)
        def on_drag_update(_, offset_x, offset_y):
            self.on_drag_update(offset_x, offset_y)
        def on_motion(_, x, y):
            self.on_motion(x, y)
        self.gesture_drag.connect("drag_begin", on_drag_begin)
        self.gesture_drag.connect("drag_update", on_drag_update)
        self.add_controller(self.gesture_drag)

        event_controller_motion = Gtk.EventControllerMotion.new()
        event_controller_motion.connect("motion", on_motion)
        self.add_controller(event_controller_motion)

        self.style_manager = None

    def update_duration(self, value):
        self._duration = value
        self._position = 0
        self.queue_draw()

    def update_position(self, value):
        self._position = value
        self.queue_draw()

    def on_drag_update(self, offset_x, offset_y):
        x = self.drag_start + offset_x
        x = max(0, x)
        allocation = self.get_allocation()
        width = allocation.width
        new_position = int((x / width) * self._duration)
        self.update_position(new_position)
        self.emit('seek_requested', new_position)

    def on_drag_begin(self, x, y):
        self.drag_start = x
        self.on_drag_update(0, 0)

    def on_motion(self, x, y):
        pass

    def do_snapshot(self, s):
        """
        AFAIK, it's currently only possible to get the accent color programmatically. Other colors need apparently be to be
        hardcoded and pray that it somewhat matches the documented Adwaita colors (unless another theme is used)
        https://gnome.pages.gitlab.gnome.org/libadwaita/doc/main/css-variables.html
        """
        allocation = self.get_allocation()
        width = allocation.width
        height = allocation.height

        position_x = int((self._position / self._duration) * width) if self._duration > 0 else 0

        if self.style_manager:
            accent_color = self.style_manager.get_accent_color()
            uses_dark_scheme = bool(self.style_manager.get_dark())
        else:
            accent_color = Adw.AccentColor.BLUE
            uses_dark_scheme = False

        colour = Gdk.RGBA()
        if uses_dark_scheme:
            colour.parse("#ffffff1a")
        else:
            colour.parse("#0000001a")

        color_position = accent_color.to_rgba(accent_color)

        position_width = 4
        before_width = position_x - position_width // 2
        after_width = width - position_width - before_width

        rect_height = height

        rounded_corner = Graphene.Size()
        rounded_corner.init(10, 10)
        regular_corner = Graphene.Size()
        regular_corner.init(0, 0)

        before = Graphene.Rect().init(0, 0, before_width, rect_height)
        before_rounded = Gsk.RoundedRect()
        before_rounded.init(before, rounded_corner, regular_corner, regular_corner, rounded_corner)
        s.push_rounded_clip(before_rounded)
        s.append_color(colour, before)
        s.pop()

        position = Graphene.Rect().init(before_width, 0, position_width, rect_height)
        s.append_color(color_position, position)

        after = Graphene.Rect().init(before_width + position_width, 0, after_width,rect_height)
        after_rounded = Gsk.RoundedRect()
        after_rounded.init(after, regular_corner, rounded_corner, rounded_corner, regular_corner)
        s.push_rounded_clip(after_rounded)
        s.append_color(colour, after)
        s.pop()
