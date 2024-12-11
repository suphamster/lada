import argparse
from tqdm import tqdm
import os
import tempfile
from lada import MODEL_WEIGHTS_DIR, VERSION
from lada.lib.restored_mosaic_frames_generator import load_models, FrameRestorer
from lada.lib.video_utils import get_video_meta_data, VideoWriter
from lada.lib import audio_utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='path to pixelated video file')
    parser.add_argument('--output', type=str, help='path to store restored video')
    parser.add_argument('--device', type=str, default="cuda:0", help='torch device to run the models on (default: %(default)s)')
    parser.add_argument('--max-clip-length', type=int, default=180, help='number of consecutive frames that will be fed to mosaic restoration model. Lower values reduce RAM and VRAM usage. If set too low quality will reduce / flickering (default: %(default)s)')
    parser.add_argument('--preserve-relative-scale',  default=True, action=argparse.BooleanOptionalAction, help="(default: %(default)s)")
    parser.add_argument('--version',  action='store_true', help="shows version and exit")

    export = parser.add_argument_group('Video export (Decoder settings)')
    export.add_argument('--codec', type=str, default="h264", help='FFmpeg video codec. E.g. "h264 or "hevc" (default: %(default)s)')
    export.add_argument('--crf', type=int, default=20, help='constant rate factor (quality setting for decoder). The lower the better with the caveat of larger files size and CPU usage (default: %(default)s)')
    export.add_argument('--moov-front',  default=False, action=argparse.BooleanOptionalAction, help="sets ffmpeg mov flags 'frag_keyframe+empty_moov+faststart'. Enables playing the output video while it's being written (default: %(default)s)")

    group_restoration = parser.add_argument_group('Mosaic restoration')
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default="basicvsrpp-generic", help="Model used to restore mosaic clips (default: %(default)s)")
    group_restoration.add_argument('--mosaic-restoration-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.1.pth'), help="(default: %(default)s)")
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str)

    group_detection = parser.add_argument_group('Mosaic detection')
    group_detection.add_argument('--mosaic-detection-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt'), help="(default: %(default)s)")

    group_cleaning = parser.add_argument_group('Mosaic cleaning')
    group_cleaning.add_argument('--mosaic-cleaning',  default=False, action=argparse.BooleanOptionalAction, help='If enabled will clean detected mosaic pattern from noise before passing it to mosaic restoration model. Not recommended (default: %(default)s)')
    group_cleaning.add_argument('--mosaic-cleaning-edge-detection-method', type=str, default="canny", help="either canny or pidinet (default: %(default)s)")
    group_cleaning.add_argument('--mosaic-cleaning-edge-detection-model-path', type=str, default=os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_edge_detection_model.pth'), help="path to pidinet tiny model (default: %(default)s)")

    return parser.parse_args()


def cli():
    args = parse_args()
    if args.version:
        print("Lada: ", VERSION)
        exit(0)
    if not (args.input and args.output):
        print("Arguments --input and --output are required. Use --help to find out more.")
        exit(1)

    mosaic_cleaning_edge_detection_model_path = args.mosaic_cleaning_edge_detection_model_path if args.mosaic_cleaning and args.mosaic_cleaning_edge_detection_method == 'pidinet' else None

    mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, preferred_pad_mode = load_models(
        args.device, args.mosaic_restoration_model, args.mosaic_restoration_model_path, args.mosaic_restoration_config_path,
        args.mosaic_detection_model_path, mosaic_cleaning_edge_detection_model_path
    )

    video_metadata = get_video_meta_data(args.input)

    frame_restorer = FrameRestorer(args.device, args.input, args.preserve_relative_scale, args.max_clip_length, args.mosaic_restoration_model,
                 mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, preferred_pad_mode, mosaic_cleaning=args.mosaic_cleaning)

    video_tmp_file_output_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(os.path.splitext(args.output)[0])}.tmp{os.path.splitext(args.output)[1]}")
    video_writer = VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height, video_metadata.video_fps_exact, codec=args.codec, crf=args.crf, moov_front=args.moov_front, time_base=video_metadata.time_base)

    for restored_frame, restored_frame_pts in tqdm(frame_restorer(), total=video_metadata.frames_count, desc="Processing frames"):
        video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)

    video_writer.release()

    print("Processing audio")
    audio_utils.combine_audio_video_files(args.input, video_tmp_file_output_path, args.output)


if __name__ == '__main__':
    cli()
