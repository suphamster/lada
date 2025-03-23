import argparse
import pathlib

import av
import torch
from tqdm import tqdm
import os
import tempfile
from lada import MODEL_WEIGHTS_DIR, VERSION
from lada.lib.frame_restorer import load_models, FrameRestorer
from lada.lib.video_utils import get_video_meta_data, VideoWriter
from lada.lib import audio_utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='Path to pixelated video file')
    parser.add_argument('--output', type=str, help='Path to save restored video')
    parser.add_argument('--device', type=str, default="cuda:0", help='torch device to run the models on. Use "cpu" or "cuda". If you have multiple GPUs you can select a specific one via index e.g. "cuda:0" (default: %(default)s)')
    parser.add_argument('--max-clip-length', type=int, default=180, help='number of consecutive frames that will be fed to mosaic restoration model. Lower values reduce RAM and VRAM usage. If set too low quality will reduce / flickering (default: %(default)s)')
    parser.add_argument('--preserve-relative-scale',  default=True, action=argparse.BooleanOptionalAction, help="(default: %(default)s)")
    parser.add_argument('--version', action='store_true', help="Shows version")

    export = parser.add_argument_group('Video export (Encoder settings)')
    export.add_argument('--codec', type=str, default="h264", help='FFmpeg video codec. E.g. "h264, "hevc" or "hevc_nvenc". Use "--list-available-codecs" to see whats available. (default: %(default)s)')
    export.add_argument('--crf', type=int, default=21, help='Constant rate factor (quality setting for video encoder). The lower the better with the caveat of producing larger files size and increased compute resources. Note: If you have selected GPU codecs "h264_nvenc" or "hevc_nvenc" then the option "qp" will be used instead as those encoders do not support the option "crf". (default: %(default)s)')
    export.add_argument('--preset', type=str, default=None, help='Encoder preset. Mostly affects file-size and speed. (default: %(default)s)')
    export.add_argument('--moov-front',  default=False, action=argparse.BooleanOptionalAction, help="sets ffmpeg mov flags 'frag_keyframe+empty_moov+faststart'. Enables playing the output video while it's being written (default: %(default)s)")
    export.add_argument('--list-codecs', action='store_true', help="List available Codecs and hardware devices / GPUs for hardware-accelerated video encoding. Uses FFmpeg wrapper library PyAV which is used for encoding the restored video.")

    group_restoration = parser.add_argument_group('Mosaic restoration')
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default="basicvsrpp-generic", help="Model used to restore mosaic clips (default: %(default)s)")
    group_restoration.add_argument('--mosaic-restoration-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth'), help="(default: %(default)s)")
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str)

    group_detection = parser.add_argument_group('Mosaic detection')
    group_detection.add_argument('--mosaic-detection-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt'), help="(default: %(default)s)")

    return parser.parse_args()

def dump_pyav_codecs():
    print(f"PyAV version: {av.__version__}")

    from av.codec.codec import dump_codecs
    dump_codecs()

    try:
        from av.codec.hwaccel import hwdevices_available
        print("Hardware device types:")
        for x in hwdevices_available():
            print("   ", x)
    except ImportError:
        print("Unable to list available hwdevices, ImportError")

    try:
        from av.codec.codec import dump_hwconfigs
        dump_hwconfigs()
    except ImportError:
        print("Unable to list hwdevice configs, ImportError")

def main():
    args = parse_args()
    if args.version:
        print("Lada: ", VERSION)
        exit(0)
    if args.list_codecs:
        dump_pyav_codecs()
        exit(0)
    if not (args.input and args.output):
        print("Arguments --input and --output are required. Use --help to find out more.")
        exit(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"GPU {args.device} selected but CUDA is not available")
        exit(1)

    mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
        args.device, args.mosaic_restoration_model, args.mosaic_restoration_model_path, args.mosaic_restoration_config_path,
        args.mosaic_detection_model_path
    )

    video_metadata = get_video_meta_data(args.input)

    frame_restorer = FrameRestorer(args.device, args.input, args.preserve_relative_scale, args.max_clip_length, args.mosaic_restoration_model,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode)
    success = True
    video_tmp_file_output_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(os.path.splitext(args.output)[0])}.tmp{os.path.splitext(args.output)[1]}")
    pathlib.Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    try:
        frame_restorer.start()

        video_writer = VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height, video_metadata.video_fps_exact, codec=args.codec, crf=args.crf, moov_front=args.moov_front, time_base=video_metadata.time_base, preset=args.preset)
        try:
            for elem in tqdm(frame_restorer, total=video_metadata.frames_count, desc="Processing frames"):
                if elem is None:
                    success = False
                    print("Error on export: frame restorer stopped prematurely")
                    break
                (restored_frame, restored_frame_pts) = elem
                video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
        finally:
            video_writer.release()
    except (Exception, KeyboardInterrupt) as e:
        success = False
        if isinstance(e, KeyboardInterrupt):
            print("Ctrl-C, stop currently running restore")
        else:
            print("Error on export", e)
    finally:
        frame_restorer.stop()

    if success:
        print("Processing audio")
        audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, args.output)
    else:
        if os.path.exists(video_tmp_file_output_path):
            os.remove(video_tmp_file_output_path)


if __name__ == '__main__':
    main()
