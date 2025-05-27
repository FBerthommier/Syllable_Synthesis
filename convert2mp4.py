from moviepy.editor import VideoFileClip


def convert_to_mp4(input_file, output_file):
    # Charger la vidéo
    clip = VideoFileClip(input_file)

    # Réencoder en MP4 (H.264 + AAC) avec MoviePy uniquement
    clip.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )


if __name__ == "__main__":
    convert_to_mp4("audio_video.avi", "audio_video.mp4")
