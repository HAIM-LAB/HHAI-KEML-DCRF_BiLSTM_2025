import librosa
import numpy as np
#import audio2raw
#from pydub import effects
#from pydub import AudioSegment
#import LoadAudio as la
#import ziptest as bz
import sys
import librosa.display
from sklearn.preprocessing import StandardScaler
import DataAugmentation as dag
import AudioProcess as ap

#import parselmouth
#from pyAudioAnalysis import audioFeatureExtraction as aF
#from pyAudioAnalysis import audioFeatureExtraction
# Convert warnings into exceptions
import warnings
#warnings.simplefilter("error")
from main import EXPECTED_FEATURES as EXPECTED_P_FEAT
#EXPECTED_P_FEAT = 190 #162
'''
*MFCCs, MFCC delta, and delta-delta: These are key features for speech and emotion detection.
*Chroma features: These are related to harmony and can be useful for emotional expression in music or speech.
*Spectral Contrast and Centroid: These features capture timbral texture, which is often linked to emotion in voice.
*Zero-crossing rate (zerocr): This is often used to understand speech/music characteristics and noise.
* Pitch: The fundamental frequency (F0) can be very important in emotion detection, as it often varies with emotional state.
* Energy: The energy or loudness of the signal over time can also be an important indicator of emotional content.
* Voice Quality Features: Features like jitter (variability in pitch) and shimmer (variability in amplitude) are also important for emotion detection in speech.
* librosa.cqt computes the Constant-Q Transform (CQT) of the audio signal X at the specified sample rate sr.
  CQT is a time-frequency representation where the frequency bins are spaced according to a logarithmic scale (like the human ear perceives sound). It is particularly effective for representing musical content, since it preserves pitch relationships more accurately than the linear Short-Time Fourier Transform (STFT), especially for lower frequencies.
* Chroma CQT captures the tonal characteristics of the speech, such as pitch classes, which are crucial for emotion detection. Emotions like happiness often correlate with higher pitch, while sadness can correspond to lower pitch.
  The CQT itself also helps in capturing the musicality and vocal pitch variations that may signal different emotional states.
* Mel spectrogram and Tonnetz are highly useful for emotion detection because they capture important aspects of timbre and tonal qualities in speech, which vary significantly with emotion.
* Mel spectrogram focuses on the overall energy distribution over different frequency bands, while Tonnetz focuses on tonal structure and harmonic relations, both of which are important for detecting emotions in speech.

'''

'''
* win_length: The length of the window used for the Short-Time Fourier Transform (STFT), or in the case of features like CQT, the number of samples per window. The window size determines the frequency resolution of the feature extraction.

	=>A larger window (e.g., 1024 samples) gives better frequency resolution but at the cost of temporal resolution.
	=>A smaller window (e.g., 400 samples) gives better temporal resolution but sacrifices some frequency resolution.
* hop_length: The number of samples between successive frames, or how much the window "slides" with each step. This affects the temporal resolution.

	=>A smaller hop_length (e.g., 160 samples) provides higher temporal resolution but will result in more frames (more data points).
	=>A larger hop_length (e.g., 512 samples) reduces the number of frames and provides lower temporal resolution.
#def generate_feature(orgin_wav, orgin_sr, win_length=400, hop_length=512, n_fft=512):
'''
def reducedFeaturesByAverage(features, n_values):
    total_bins = features.shape[0]  # Total number of pitch bins
    part_size = total_bins // n_values  # Size of each part (integer division)
    remainder = total_bins % n_values  # Remainder (leftover bins)

    # List to store the average pitch values for each part
    averaged_features = []

    start_idx = 0
    for i in range(n_values):
        # If there are remaining values, add them to the current part
        if i < remainder:
            end_idx = start_idx + part_size + 1  # Add 1 extra bin to this part
        else:
            end_idx = start_idx + part_size

        # Take the average over the bins in this part
        avg_pitch = np.mean(features[start_idx:end_idx], axis=0)
        averaged_features.append(avg_pitch)

        # Move the start index for the next part
        start_idx = end_idx

    # Convert to a numpy array (optional, depending on use case)
    return np.array(averaged_features)

# some pitch related features
def pitch_features(pitches, magnitudes):
        p_feat = np.array([])
        m_pitch = np.mean(pitches, axis=0)  # Take the mean pitch for the entire audio segment
        # Downsample to a fixed length (e.g., 10 values)
        #use m_pitch all data: is good if deep learning for large data
        pitch = reducedFeaturesByAverage(m_pitch, 12)
        p_feat = np.hstack((p_feat,pitch))
        pitchmean = np.mean(m_pitch)
        pitchstd = np.std(m_pitch)
        pitchmax = np.max(m_pitch)
        p_feat = np.hstack([p_feat, pitchmean, pitchstd, pitchmax])
        
        return p_feat

def extract_features_withsr(orgin_wav, new_sr, n_fft=512, hop_length=160, win_length=512):
    """
    Extracts various audio features for emotion detection.

    :param audio: Audio signal (numpy array)
    :param sr: Sample rate of the audio signal
    :param n_fft: FFT window size
    :param hop_length: Hop length for STFT
    :param win_length: Window length for STFT
    :return: A numpy array of extracted features
    """

    result = np.array([])

    audio = orgin_wav
    # Ensure n_fft is within valid range
    # print(len(audio))
    # n_fft = min(n_fft, len(audio))
    hop_length = n_fft // 4  # Keep reasonable overlap

    # --- 1. MFCCs and their deltas ---
    fmax = new_sr / 2  # Ensure fmax is within the valid range
    mffcFeat = 20 #20 #15
    mfccs = librosa.feature.mfcc(y=audio, sr=new_sr, n_mfcc=mffcFeat, n_fft=n_fft, hop_length=hop_length, n_mels=40, fmax=fmax, htk=True)
    # Check if MFCCs are empty or contain only zeros
    if mfccs.size == 0 or np.all(mfccs == 0):
        print("Empty frequency set found by mfccs")
        return result
    mfccs_delta = np.mean(librosa.feature.delta(mfccs,order=1).T, axis=0)
    mfccs_delta_2 = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
    mfccsstd = np.std(mfccs.T, axis=0)
    mfccs = np.mean(mfccs.T, axis=0)
    # --- 2. Magnitude spectrogram-based features ---
    linear_spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    mag, _ = librosa.magphase(linear_spectrogram)  # Get magnitude
    mag_T = mag.T  # Transpose for feature extraction

    # Get the shape of mag_T
    time_frames, freq_bins = mag_T.shape  # (number of time frames, number of frequency bins)

    mag_fft = n_fft//4

    nChroma = 12 #12 for 190 #while 100 data it was 6
    # Chroma features
    chroma_stft = np.mean(np.abs(librosa.feature.chroma_stft(S=mag_T, sr=new_sr, n_chroma=nChroma, n_fft=mag_fft, hop_length=hop_length, win_length=win_length)).T, axis=0)
    # Check if Chroma STFT is empty or contains only zeros
    if chroma_stft.size == 0 or np.all(chroma_stft == 0):
        print("Empty frequency set found by chroma_stft")
        return result

    # Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=mag_T, sr=new_sr, n_bands=5, n_fft=mag_fft, hop_length=hop_length, win_length=win_length).T,axis=0)
    
    # Spectral centroid
    #centroid = np.mean(librosa.feature.spectral_centroid(S=mag_T, sr=new_sr).T,axis=0)
    cent = librosa.feature.spectral_centroid(y=mag_T, sr=new_sr, n_fft=mag_fft)
    
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)
    centroid = np.array([meancent, stdcent,maxcent])

    # --- 3. Pitch and Energy ---
    m_pitch, magnitudes = librosa.core.piptrack(y=audio, sr=new_sr,n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # Get pitch
    
    pitch = pitch_features(m_pitch, magnitudes)
    
    ## Root mean square energy    
    rmse = librosa.feature.rms(y=audio)
    
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)
    energy = np.array([meanrms,stdrms,maxrms])
    # --- 4. Voice Quality Features 
    zerocr = np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=1024, hop_length=hop_length).T, axis=0)
    
    # --- 5. Constant-Q Transform and Chroma CQT ---
    fmin1 = librosa.midi_to_hz(24)  # MIDI note 24 corresponds to C1 (32.7 Hz)
    cqt = np.abs(librosa.cqt(audio, sr=new_sr, hop_length=hop_length, fmin=fmin1))
    
    #n_chroma=12, bins_per_octave=48  is good if deep learning for large data
    chroma_cqt = np.mean(librosa.feature.chroma_cqt(C=cqt, sr=new_sr, hop_length=hop_length, n_chroma=nChroma, n_octaves=7, bins_per_octave=36).T,axis=0)
    
    chroma_cens = librosa.feature.chroma_cens(C=cqt, sr=new_sr,hop_length=hop_length, n_chroma=nChroma, n_octaves=7, bins_per_octave=36)
    
    chroma_cens = np.mean(chroma_cens.T, axis=0);

    # --- 6. Mel spectrogram ---
    #n_mels=128 is good if deep learning for large data
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=new_sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=64) 
    
    log_mel_spec = np.mean(librosa.power_to_db(mel_spec).T,axis=0)  # Convert power to dB scale

    if(mfccs.shape[0] != mffcFeat):    
        print("mfccs.shape='{}', dim='{}'".format(mfccs.shape,mfccs.ndim))
    if(mfccs_delta.shape[0] != mffcFeat):    
        print("mfccs_delta.shape='{}', dim='{}'".format(mfccs_delta.shape,mfccs_delta.ndim))
    if(mfccs_delta_2.shape[0] != mffcFeat):
        print("mfccs_delta2.shape='{}', dim='{}'".format(mfccs_delta_2.shape,mfccs_delta_2.ndim))
    if(mfccsstd.shape[0] != mffcFeat):
        print("mfccsstd.shape='{}', dim='{}'".format(mfccsstd.shape,mfccsstd.ndim))
    if(chroma_stft.shape[0] != nChroma):
        print("chroma_stft.shape='{}', dim='{}'".format(chroma_stft.shape,chroma_stft.ndim))
    if(contrast.shape[0] != 6):
        print("contrast.shape='{}', dim='{}'".format(contrast.shape,contrast.ndim))
    if(pitch.shape[0] != 15):
        print("pitch.shape='{}', dim='{}'".format(pitch.shape,pitch.ndim))
    if(chroma_cqt.shape[0] != nChroma):
        print("chroma_cqt.shape='{}', dim='{}'".format(chroma_cqt.shape,chroma_cqt.ndim))
    if(chroma_cens.shape[0] != nChroma):
        print("chroma_cens.shape='{}', dim='{}'".format(chroma_cens.shape,chroma_cens.ndim))    
    if(log_mel_spec.shape[0] != 64): #64): #10):
        print("log_mel_spec.shape='{}', dim='{}'".format(log_mel_spec.shape,log_mel_spec.ndim))
    if(centroid.shape[0] != 3):
        print("centroid.shape='{}', dim='{}'".format(centroid.shape,centroid.ndim))
    if(energy.shape[0] != 3):
        print("energy.shape='{}', dim='{}'".format(energy.shape,energy.ndim))
    if(zerocr.shape[0] != 1):
        print("zerocr.shape='{}', dim='{}'".format(zerocr.shape,zerocr.ndim))
    feat = np.hstack([
        mfccs, mfccs_delta, mfccs_delta_2, mfccsstd,
        chroma_stft, chroma_cqt, chroma_cens,
        log_mel_spec, contrast,
        energy, zerocr
        ])
    return feat
def extract_features(filepath, orgin_wav, orgin_sr):
    
    temp_result = np.array([])
    new_sr = 22050 #orgin_sr
    audio_data, length = ap.process_wav(orgin_wav, orgin_sr, new_sr)
    if length < 0.50:
        print("Small length audio found")
        print(length)
        print(filepath)
        return temp_result;
    n_fft=512
    hop_length=160
    win_length=512

    feat1 = extract_features_withsr(audio_data, new_sr, n_fft, hop_length, win_length)
    if len(feat1) != EXPECTED_P_FEAT:
        return temp_result
    # data with noise
    noise_data = dag.noise2wav(audio_data)
    feat2 = extract_features_withsr(noise_data, new_sr, n_fft, hop_length, win_length)
    if len(feat2) != EXPECTED_P_FEAT:
        return temp_result
    
    noise_data = dag.noise2wav1(audio_data)
    feat21 = extract_features_withsr(noise_data, new_sr, n_fft, hop_length, win_length)
    if len(feat21) != EXPECTED_P_FEAT:
        return temp_result

    noise_data = dag.noise2wav2(audio_data)
    feat22 = extract_features_withsr(noise_data, new_sr, n_fft, hop_length, win_length)
    if len(feat22) != EXPECTED_P_FEAT:
        return temp_result

    # data with stretching and pitching
    new_data = dag.stretch_wav(audio_data)
    data_stretch_pitch = dag.pitch_wav(new_data, new_sr)
    feat3 = extract_features_withsr(data_stretch_pitch, new_sr, n_fft, hop_length, win_length)
    if len(feat3) != EXPECTED_P_FEAT:
        return temp_result

    new_data = dag.stretch_wav1(audio_data)
    data_stretch_pitch = dag.pitch_wav1(new_data, new_sr)
    feat31 = extract_features_withsr(data_stretch_pitch, new_sr, n_fft, hop_length, win_length)
    if len(feat31) != EXPECTED_P_FEAT:
        return temp_result

    new_data = dag.stretch_wav2(audio_data)
    data_stretch_pitch = dag.pitch_wav2(new_data, new_sr)
    feat32 = extract_features_withsr(data_stretch_pitch, new_sr, n_fft, hop_length, win_length)
    if len(feat31) != EXPECTED_P_FEAT:
        return temp_result

    result = np.vstack([feat1, feat2, feat21, feat22, feat3, feat31, feat32])

    return result
