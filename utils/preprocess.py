import librosa
import numpy as np

def load_audio(file, sr=16000):
    audio, _ = librosa.load(file, sr=sr)
    return audio

def create_mel(audio, sr=16000, n_mels=64):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=256,
        n_fft=1024
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
