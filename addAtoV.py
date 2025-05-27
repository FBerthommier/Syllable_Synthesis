from moviepy.editor import VideoFileClip, AudioClip
import librosa
import numpy as np
import soundfile as sf

def main():

   # Charger en mono à 20 kHz
    audio_data, sr = librosa.load("essai.wav", sr=20000)

    # Effectuer le resampling de 20000 Hz à 40000 Hz avec librosa
    resampled_audio = librosa.resample(audio_data, orig_sr=20000, target_sr=40000)

     
    # Fonction pour convertir le tableau numpy en AudioClip
    def make_frame(t):
         # Si t est un scalaire, traiter comme un index unique
         if np.isscalar(t):
             idx = int(t * 40000)  # Convertir le temps en index pour un scalaire
             return [resampled_audio[idx]] if idx < len(resampled_audio) else [0]
         else:
             # Si t est un tableau, convertir chaque élément en index
             idx = (np.array(t) * 40000).astype(int)  # Convertir le temps en index pour un tableau
             # Retourner les échantillons audio correspondants
             return np.clip(resampled_audio[idx], a_min=-32768, a_max=32767).tolist()
 
    # Créer un AudioClip à partir du tableau numpy en utilisant make_frame
    resampled_audio_clip = AudioClip(make_frame, duration=len(resampled_audio)/40000, fps=40000)

    # Charger la vidéo à laquelle vous voulez ajouter l'audio
    video_clip = VideoFileClip("output.avi")

    # Ajouter l'audio rééchantillonné à la vidéo
    final_clip = video_clip.set_audio(resampled_audio_clip)

    # Sauvegarder le fichier vidéo avec l'audio
    final_clip.write_videofile("audio_video.avi", codec="libx264", audio_codec="aac")

if __name__ == "__main__":
        main()