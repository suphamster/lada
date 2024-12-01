import os
import subprocess
import shutil
from typing import Optional

def combine_audio_video_files(av_video_input_path, tmp_v_video_input_path, av_video_output_path):
    audio_codec = get_audio_codec(av_video_input_path)
    if audio_codec:
        needs_reencoding = not is_output_container_compatible_with_input_audio_codec(audio_codec, av_video_output_path)
        if needs_reencoding:
            os.system("ffmpeg -y -loglevel quiet -i '%s' -i '%s' -c:v copy -map 0:v:0 -map 1:a:0 '%s'" % (tmp_v_video_input_path, av_video_input_path, av_video_output_path))
        else:
            os.system("ffmpeg -y -loglevel quiet -i '%s' -i '%s' -c copy -map 0:v:0 -map 1:a:0 '%s'" % (tmp_v_video_input_path, av_video_input_path, av_video_output_path))
        os.remove(tmp_v_video_input_path)
    else:
        shutil.move(tmp_v_video_input_path, av_video_output_path)

def get_audio_codec(file_path: str) -> Optional[str]:
    cmd = f"ffprobe -loglevel error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1"
    cmd = cmd.split() + [file_path]
    cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE)
    audio_codec = cmd_result.stdout.decode('utf-8').strip().lower()
    return audio_codec if len(audio_codec) > 0 else None

def is_output_container_compatible_with_input_audio_codec(audio_codec, output_path):
    file_extension = os.path.splitext(output_path)[1]
    known_container_file_extensions = ('.mp4', '.mkv')
    known_compatible_audio_codecs = ('aac', 'mp3', 'opus')
    return file_extension in known_container_file_extensions and audio_codec in known_compatible_audio_codecs