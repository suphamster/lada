import pathlib
import sys
import gi

from lada import VERSION

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')

from gi.repository import Gtk, Gio, Adw, Gdk, Gst

here = pathlib.Path(__file__).parent.resolve()

from lada.gui.window import MainWindow
from lada.gui.shortcuts import ShortcutsWindow, ShortcutsManager

class LadaApplication(Adw.Application):

    def __init__(self):
        super().__init__(application_id='io.github.ladaapp.lada',
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.create_action('quit', lambda *_: self.quit(), ['<primary>q'])
        self.create_action('about', self.on_about_action)
        self.create_action('shortcuts', self.on_shortcut_action)

        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(str(here.joinpath('style.css')))
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        resource = Gio.resource_load(str(here.joinpath('resources.gresource')))
        Gio.resources_register(resource)

        self.shortcuts = ShortcutsManager(self)

        Gst.init(None)

    def do_activate(self):
        win = self.props.active_window
        if not win:
            win = MainWindow(application=self)
            self.shortcuts.init(win.shortcut_controller)
        win.present()

    def on_about_action(self, *args):
        about = Adw.AboutDialog(application_name='lada',
                                application_icon='io.github.ladaapp.lada',
                                version=VERSION)
        about.present(self.props.active_window)

    def on_shortcut_action(self, *args):
        shortcuts_window = ShortcutsWindow(self)
        shortcuts_window.show()

    def create_action(self, name, callback, shortcuts=None):
        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.add_action(action)
        if shortcuts:
            self.set_accels_for_action(f"app.{name}", shortcuts)

def main():
    app = LadaApplication()
    return app.run(sys.argv)

if __name__ == "__main__":
    main()
