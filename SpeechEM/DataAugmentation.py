import librosa
import numpy as np

def noise2wav(wav):
    noise_it = 0.035
    noise_amp = noise_it*np.random.uniform()*np.amax(wav)
    wav = wav + noise_amp*np.random.normal(size=wav.shape[0])
    return wav

def noise2wav1(wav):
    noise_it = 0.025
    noise_amp = noise_it*np.random.uniform()*np.amax(wav)
    wav = wav + noise_amp*np.random.normal(size=wav.shape[0])
    return wav

def noise2wav2(wav):
    noise_it = 0.015
    noise_amp = noise_it*np.random.uniform()*np.amax(wav)
    wav = wav + noise_amp*np.random.normal(size=wav.shape[0])
    return wav

def stretch_wav(wav):
    str_rate=0.8
    return librosa.effects.time_stretch(wav, rate=str_rate)

def stretch_wav1(wav):
    str_rate=0.9
    return librosa.effects.time_stretch(wav, rate=str_rate)

def stretch_wav2(wav):
    str_rate=0.7
    return librosa.effects.time_stretch(wav, rate=str_rate)

def shift_wav(wav):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(wav, shift_range)

def shift_wav1(wav):
    shift_range = int(np.random.uniform(low=-4, high = 4)*1000)
    return np.roll(wav, shift_range)

def shift_wav2(wav):
    shift_range = int(np.random.uniform(low=-3, high = 3)*1000)
    return np.roll(wav, shift_range)

def pitch_wav(wav, w_sr):
    steps=0.70
    return librosa.effects.pitch_shift(wav, sr=w_sr, n_steps=steps)

def pitch_wav1(wav, w_sr):
    steps=0.80
    return librosa.effects.pitch_shift(wav, sr=w_sr, n_steps=steps)
def pitch_wav2(wav, w_sr):
    steps=0.70
    return librosa.effects.pitch_shift(wav, sr=w_sr, n_steps=steps)
