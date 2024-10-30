import os
import pathlib
import tempfile
import threading

from gi.repository import Gtk, GObject, GdkPixbuf, GLib, Gio, Gst, GstApp, Adw

from lada.gui.timeline import Timeline
from lada.lib import audio_utils
from lada.lib.restored_mosaic_frames_generator import load_models, FrameRestorer
from lada.lib.video_utils import get_video_meta_data, VideoMetadata, VideoWriter
from lada import MODEL_WEIGHTS_DIR

here = pathlib.Path(__file__).parent.resolve()


@Gtk.Template(filename=here / 'video_preview.ui')
class VideoPreview(Gtk.Widget):
    __gtype_name__ = 'VideoPreview'

    button_play_pause = Gtk.Template.Child()
    picture_video_preview = Gtk.Template.Child()
    widget_timeline: Timeline = Gtk.Template.Child()
    button_image_play_pause = Gtk.Template.Child()
    label_current_time = Gtk.Template.Child()
    spinner_video_preview = Gtk.Template.Child()


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._passthrough = False
        self._mosaic_cleaning = False
        self._mosaic_detection = False
        self._mosaic_restoration_model = 'basicvsrpp-generic'
        self.frame_restorer = None
        self.frame_restorer_generator = None
        self._video_file_ready = False
        self._max_clip_length = 180
        self.appsrc = None
        self.audio_uridecodebin = None
        self.pipeline = None
        self.file_duration_ns = 0
        self.file_duration_frames = 0
        self._buffer_queue_min_thresh_time = 14
        self.file_path = None
        self.frame_num = 0
        self.frame_duration_ns = None
        self.video_metadata: VideoMetadata | None = None
        self.models_cache = None
        self._device = "cuda:0"
        self.video_buffer_queue = None
        self.audio_buffer_queue = None
        self._application = None
        self.should_be_paused = False

        self.widget_timeline.connect('seek_requested', lambda widget, seek_position: self.on_seek_requested(seek_position))

    @GObject.Property()
    def passthrough(self):
        return self._passthrough

    @passthrough.setter
    def passthrough(self, value):
        self._passthrough = value
        self.on_seek_requested(self.frame_num)

    @GObject.Property()
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self.models_cache = None
        self.on_seek_requested(self.frame_num)

    @GObject.Property()
    def max_clip_length(self):
        return self._max_clip_length

    @max_clip_length.setter
    def max_clip_length(self, value):
        self._max_clip_length = value
        self.on_seek_requested(self.frame_num)

    @GObject.Property()
    def buffer_queue_min_thresh_time(self):
        return self._buffer_queue_min_thresh_time

    @buffer_queue_min_thresh_time.setter
    def buffer_queue_min_thresh_time(self, value):
        self._buffer_queue_min_thresh_time = value

        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2

        self.video_buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)
        self.video_buffer_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)
        self.audio_buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)
        self.audio_buffer_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)

    @GObject.Property()
    def mosaic_restoration_model(self):
        return self._mosaic_restoration_model

    @mosaic_restoration_model.setter
    def mosaic_restoration_model(self, value):
        assert value in ['basicvsrpp-generic', 'basicvsrpp-bj-pov', 'deepmosaics'], f"only 'basicvsrpp-generic', 'basicvsrpp-bj-pov' and 'deepmosaics' restoration models currently supported but received {value}"
        self._mosaic_restoration_model = value
        self.on_seek_requested(self.frame_num)

    @GObject.Property()
    def mosaic_cleaning(self):
        return self._mosaic_cleaning

    @mosaic_cleaning.setter
    def mosaic_cleaning(self, value):
        self._mosaic_cleaning = value
        self.on_seek_requested(self.frame_num)

    @GObject.Property()
    def mosaic_detection(self):
        return self._mosaic_detection

    @mosaic_detection.setter
    def mosaic_detection(self, value):
        self._mosaic_detection = value
        self.on_seek_requested(self.frame_num)

    @GObject.Property(type=Adw.Application)
    def application(self):
        return self._application

    @application.setter
    def application(self, value):
        self._application = value

    @GObject.Signal(name="video_file_ready")
    def video_file_ready_signal(self):
        pass

    @GObject.Signal(name="video_export_finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video_export_progress")
    def video_export_progress_signal(self, status: float):
        pass

    @Gtk.Template.Callback()
    def button_play_pause_callback(self, button_clicked):
        pipe_state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        if pipe_state.state == Gst.State.PLAYING:
            self.should_be_paused = True
            self.pipeline.set_state(Gst.State.PAUSED)
        elif pipe_state.state == Gst.State.PAUSED:
            self.should_be_paused = False
            self.pipeline.set_state(Gst.State.PLAYING)
        else:
            print("unhandled pipeline state in button_play_pause_callback", pipe_state.nick_value)

    def on_seek_requested(self, seek_position):
        if self.pipeline:
            (res, current_position_ns) = self.pipeline.query_position(Gst.Format.TIME)
            seek_position_ns = int(seek_position * self.file_duration_ns / self.file_duration_frames)
            print(f"try to seek from {int(current_position_ns / Gst.SECOND)} (sec) to {int(seek_position_ns / Gst.SECOND)} (sec)")
            success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_position_ns)
            print(f"seeking return val:", success)


    def open_video_file_gstreamer(self, file):
        self.file_path = file.get_path()
        self.video_metadata = get_video_meta_data(self.file_path)

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self.file_duration_frames = self.video_metadata.frames_count

        self.widget_timeline.set_property("duration", self.file_duration_frames)

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self._video_file_ready = False
            self.frame_restorer = None
            self.frame_restorer_generator = None
            self.adjust_pipeline_with_new_source_file()
        else:
            self.init_pipeline()

        self.pipeline.set_state(Gst.State.PLAYING)

    def export_video(self, file_path, video_codec, crf):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self._video_file_ready = False
        self.setup_frame_restorer(start_frame=0)
        self.frame_restorer_generator = self.frame_restorer()

        def run_export():
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(file_path)[0])}.tmp{os.path.splitext(file_path)[1]}")
            video_writer = VideoWriter(video_tmp_file_output_path, self.video_metadata.video_width,
                                       self.video_metadata.video_height, self.video_metadata.video_fps_exact,
                                       codec=video_codec, crf=crf)

            progress_update_step_size = 100
            for frame_num, restored_frame in enumerate(self.frame_restorer_generator):
                video_writer.write(restored_frame, bgr2rgb=True)
                if frame_num % progress_update_step_size == 0:
                    self.emit('video_export_progress', frame_num / self.video_metadata.frames_count)

            video_writer.release()

            audio_utils.combine_audio_video_files(self.file_path, video_tmp_file_output_path, file_path)
            self.emit('video_export_finished')

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()

    def adjust_pipeline_with_new_source_file(self):
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.video_metadata.video_width},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")
        self.appsrc.set_property('caps', caps)
        self.appsrc.set_property('duration', self.file_duration_ns)
        self.audio_uridecodebin.set_property('uri', 'file://' + self.file_path)

    def autoplay_if_enough_data_buffered(self):
        if not self.should_be_paused:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.spinner_video_preview.set_visible(False)

    def autopause_if_not_enough_data_buffered(self):
        self.pipeline.set_state(Gst.State.PAUSED)
        self.spinner_video_preview.set_visible(True)

    def init_pipeline(self):
        pipeline = Gst.Pipeline.new()

        appsrc = Gst.ElementFactory.make('appsrc', "numpy-source")
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.video_metadata.video_width},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")
        appsrc.set_property('caps', caps)
        appsrc.set_property('is-live', False)
        appsrc.set_property('emit-signals', True)
        appsrc.set_property('stream-type', GstApp.AppStreamType.SEEKABLE)
        appsrc.set_property('format', Gst.Format.TIME)
        appsrc.set_property('duration', self.file_duration_ns)
        appsrc.connect('need-data', self.on_need_data)
        appsrc.connect('seek-data', self.on_seek_data)
        def on_eos(appsrc):
            self.autoplay_if_enough_data_buffered()
            return True
        appsrc.connect("end-of-stream", on_eos)
        pipeline.add(appsrc)

        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2

        buffer_queue = Gst.ElementFactory.make('queue', None)
        buffer_queue.set_property('max-size-bytes', 0)
        buffer_queue.set_property('max-size-buffers', 0)
        buffer_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)  # ns
        buffer_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)
        def on_running(queue):
            current_level_time = queue.get_property('current-level-time')
            if not self._video_file_ready and current_level_time > 0:
                self._video_file_ready = True
                self.emit('video_file_ready')
        buffer_queue.connect("underrun", lambda queue: self.autopause_if_not_enough_data_buffered())
        buffer_queue.connect("overrun", lambda queue: self.autoplay_if_enough_data_buffered())
        buffer_queue.connect('running', on_running)
        pipeline.add(buffer_queue)

        audio_queue = Gst.ElementFactory.make('queue', None)
        audio_queue.set_property('max-size-bytes', 0)
        audio_queue.set_property('max-size-buffers', 0)
        audio_queue.set_property('max-size-time', buffer_queue_max_thresh_time * Gst.SECOND)  # ns
        audio_queue.set_property('min-threshold-time', buffer_queue_min_thresh_time * Gst.SECOND)
        pipeline.add(audio_queue)

        audio_uridecodebin = Gst.ElementFactory.make('uridecodebin', None)
        audio_uridecodebin.set_property('uri', 'file://' + self.file_path)
        def on_pad_added(decodebin, decoder_src_pad, audio_queue):
            caps = decoder_src_pad.get_current_caps()
            if not caps:
                caps = decoder_src_pad.query_caps()
            gststruct = caps.get_structure(0)
            gstname = gststruct.get_name()
            if gstname.startswith("audio"):
                sink_pad = audio_queue.get_static_pad("sink")
                decoder_src_pad.link(sink_pad)
        audio_uridecodebin.connect("pad-added", on_pad_added, audio_queue)
        pipeline.add(audio_uridecodebin)

        audio_audioconvert = Gst.ElementFactory.make('audioconvert', None)
        pipeline.add(audio_audioconvert)

        audio_sink = Gst.ElementFactory.make('autoaudiosink', None)
        pipeline.add(audio_sink)

        gtksink = Gst.ElementFactory.make('gtk4paintablesink', None)
        paintable = gtksink.get_property('paintable')
        if paintable.props.gl_context:
            video_sink = Gst.ElementFactory.make('glsinkbin', None)
            video_sink.set_property('sink', gtksink)
        else:
            video_sink = Gst.Bin.new()
            convert = Gst.ElementFactory.make('videoconvert', None)
            video_sink.add(convert)
            video_sink.add(gtksink)
            convert.link(gtksink)
            video_sink.add_pad(Gst.GhostPad.new('sink', convert.get_static_pad('sink')))
        pipeline.add(video_sink)

        # note that we cannot link decodebin directly to audio_queue as pads are dynamically added and not available at this point
        # see on_pad_added()
        audio_queue.link(audio_audioconvert)
        audio_audioconvert.link(audio_sink)

        appsrc.link(buffer_queue)
        buffer_queue.link(video_sink)

        def on_bus_msg(_, msg):
            match msg.type:
                case Gst.MessageType.EOS:
                    print("eos")
                case Gst.MessageType.ERROR:
                    (err, _) = msg.parse_error()
                    print(f'Error from {msg.src.get_path_string()}: {err}')
                case Gst.MessageType.STATE_CHANGED:
                    old_state, new_state, pending_state = msg.parse_state_changed()
                    if old_state == Gst.State.PAUSED and new_state == Gst.State.PLAYING:
                        self.button_image_play_pause.set_property("icon-name", "media-playback-pause-symbolic")
                    elif old_state == Gst.State.PLAYING and new_state == Gst.State.PAUSED:
                        self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
                case Gst.MessageType.STREAM_STATUS:
                    pass
                case _:
                    # print("other message", msg.type)
                    pass
            return True

        bus = pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, on_bus_msg)

        self.appsrc = appsrc
        self.audio_uridecodebin = audio_uridecodebin
        self.pipeline = pipeline
        self.video_buffer_queue, self.audio_buffer_queue = buffer_queue, audio_queue
        self.picture_video_preview.set_paintable(paintable)

        GLib.timeout_add(500, self.update_current_position)

    def on_seek_data(self, appsrc, offset_ns):
        self.pipeline.set_state(Gst.State.PAUSED)
        seek_position_frames = int(offset_ns * self.file_duration_frames / self.file_duration_ns)
        print(f"called on_seek_data of appsrc with offset (sec): {offset_ns / Gst.SECOND} (frame: {seek_position_frames})")
        if self.frame_restorer:
            self.setup_frame_restorer(start_frame=seek_position_frames)
            self.frame_restorer_generator = self.frame_restorer()
        return True

    def on_need_data(self, src, length):
        if not self.frame_restorer:
            self.setup_frame_restorer(start_frame=0)
            self.frame_restorer_generator = self.frame_restorer()

        if self.frame_num < self.video_metadata.frames_count:
            frame = next(self.frame_restorer_generator)
            data = frame.tostring()

            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.frame_duration_ns
            timestamp = self.frame_num * self.frame_duration_ns
            buf.pts = int(timestamp)
            buf.offset = self.frame_num
            src.emit('push-buffer', buf)
            self.frame_num += 1
        else:
            src.emit("end-of-stream")
            print("reached end of stream")

    def update_current_position(self):
        res, position = self.pipeline.query_position(Gst.Format.TIME)
        if not res or position == -1:
            self.label_current_time.set_text('00:00:00')
        else:
            seconds = int(position / Gst.SECOND)
            minutes = int(seconds / 60)
            hours = int(minutes / 60)
            seconds = seconds % 60
            minutes = minutes % 60
            hours, minutes, seconds = int(hours), int(minutes), int(seconds)
            time = f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"
            self.label_current_time.set_text(time)
            position_frames = int(position * self.file_duration_frames / self.file_duration_ns)
            self.widget_timeline.set_property("position", position_frames)

        return True


    def setup_frame_restorer(self, start_frame=0):
        if self._mosaic_restoration_model == 'deepmosaics':
            mosaic_restoration_model_path = os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'clean_youknow_video.pth')
        elif self._mosaic_restoration_model == 'basicvsrpp-bj-pov':
            mosaic_restoration_model_path = os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_bj_pov.pth')
        else:
            mosaic_restoration_model_path = os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic.pth')

        args = dict(
            input=self.file_path,
            output=None,
            device=self._device,
            mosaic_restoration_model=self._mosaic_restoration_model,
            mosaic_detection_model_path=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model.pt'),
            mosaic_restoration_model_path=mosaic_restoration_model_path,
            mosaic_restoration_config_path=None,
            max_clip_length=self._max_clip_length,
            clip_size=256,
            mosaic_cleaning=self._mosaic_cleaning,
            preserve_relative_scale=True
        )

        class dotdict(dict):
            """dot.notation access to dictionary attributes"""
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        args = dotdict(args)
        if self.models_cache is None or self.models_cache["mosaic_detection_model_name"] != self._mosaic_restoration_model:
            mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, mosaic_restoration_model_preferred_pad_mode = load_models(args)
            self.models_cache = dict(mosaic_detection_model_name=self._mosaic_restoration_model,
                                     mosaic_detection_model=mosaic_detection_model,
                                     mosaic_restoration_model=mosaic_restoration_model,
                                     mosaic_edge_detection_model=mosaic_edge_detection_model,
                                     mosaic_restoration_model_preferred_pad_mode=mosaic_restoration_model_preferred_pad_mode)

        self.frame_restorer = FrameRestorer(args,
                                            self.models_cache["mosaic_detection_model"],
                                            self.models_cache["mosaic_restoration_model"],
                                            self.models_cache["mosaic_edge_detection_model"],
                                            self.models_cache["mosaic_restoration_model_preferred_pad_mode"],
                                            start_frame=start_frame, passthrough=self._passthrough,
                                            mosaic_detection=self._mosaic_detection)
        self.frame_num = start_frame
