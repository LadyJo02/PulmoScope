import librosa
import numpy as np
from scipy import signal

# These mirror the notebook config
SAMPLE_RATE = 16000
DURATION = 5          # seconds target
N_MELS = 224
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

TARGET_LEN = SAMPLE_RATE * DURATION  # 80000 samples


def apply_bandpass_filter(y, lowcut=50, highcut=2500, fs=SAMPLE_RATE, order=5):
    """Remove heartbeats (<50Hz) and high-frequency hiss (>2.5kHz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, y)


def peak_normalize(y):
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def load_audio(file, sr=SAMPLE_RATE):
    """Load mono audio at the target sample rate."""
    y, _ = librosa.load(file, sr=sr)
    return y


def create_mel(y):
    """
    Recreate the training feature pipeline:

    1. Band-pass filter + peak normalize
    2. Pad/trim to fixed length
    3. Compute Mel-spectrogram (224 mels)
    4. MFCCs (40)
    5. Stack -> (264, T) for the models
    """
    # 1. filter + normalize
    y = apply_bandpass_filter(y, fs=SAMPLE_RATE)
    y = peak_normalize(y)

    # 2. pad / trim
    cur_len = len(y)
    if cur_len > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif cur_len < TARGET_LEN:
        rep = int(np.ceil(TARGET_LEN / cur_len))
        y = np.tile(y, rep)[:TARGET_LEN]

    # 3. mel
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-8)

    # 4. MFCC on top of mel
    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=N_MFCC)
    mfcc_norm = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    # 5. stack to (264, T)
    feat = np.vstack([S_norm, mfcc_norm]).astype(np.float32)
    return feat
