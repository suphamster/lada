import os
import subprocess


def combine_audio_video_files(av_video_input_path, tmp_v_video_input_path, av_video_output_path):
    cmd = f"ffprobe -loglevel error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1"
    cmd = cmd.split() + [av_video_input_path]
    cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE)
    has_audio_stream = len(cmd_result.stdout.decode('utf-8').strip()) > 0
    if has_audio_stream:
        os.system("ffmpeg -y -loglevel quiet -i '%s' -i '%s' -c copy -map 0:v:0 -map 1:a:0 '%s'" % (tmp_v_video_input_path, av_video_input_path, av_video_output_path))
        os.remove(tmp_v_video_input_path)
    else:
        os.rename(tmp_v_video_input_path, av_video_output_path)