from pydub import AudioSegment
import numpy as np
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import os, sys, requests, csv
import librosa 
import librosa.display
from IPython.display import Audio
import soundfile as sf



def outputAudio(filename, wav, sr):
	sf.write(filename, wav, sr)
def Sound2Wav(sound):
	data = sound.get_array_of_samples()
	data = np.array(data, dtype=float)
	wav = data / 32768
	return wav, sound.frame_rate	
	
def LoadWavByAudioSegment(path,oset,durations):
	sound = AudioSegment.from_file(path)
	return Sound2Wav(sound)

'''
# Read the WAV file using soundfile
wav, sr_ret = sf.read(filePath, start=int(oset * 44100), stop=int((oset + durations) * 44100))
'''
def loadAudio(filePath,oset=None,durations=None):
	wav, sr_ret = librosa.load(filePath)
	return wav, sr_ret

def remove_silence(y, sr, silence_threshold=-40.0, min_silence_length_ms=200, remove_percentage=0.7):
    """
    Removes 70% of the silence at the start and end of an audio waveform if silence is greater than 0.20 sec.
    
    Parameters:
    - y: numpy array, audio data.
    - sr: int, sample rate of the audio.
    - silence_threshold: float, dB threshold for considering silence (default is -40 dB).
    - min_silence_length_ms: int, minimum silence duration in milliseconds to be considered for trimming (default is 200 ms).
    - remove_percentage: float, percentage of silence to remove (default is 0.7 or 70%).
    
    Returns:
    - Processed audio as a numpy array.
    """
    
    # Detect the silent parts using librosa
    silent_part_start = librosa.effects.split(y, top_db=-silence_threshold)[0][0]
    silent_part_end = librosa.effects.split(y, top_db=-silence_threshold)[-1][-1]
    
    # Convert milliseconds to samples (for librosa data processing)
    silent_part_start_ms = silent_part_start / sr * 1000
    silent_part_end_ms = (len(y) - silent_part_end) / sr * 1000

    # Check if silence duration is greater than 200ms
    if silent_part_start_ms >= min_silence_length_ms:
        # Calculate the percentage to remove at the start
        silence_to_remove_start = int(silent_part_start * remove_percentage)
        y = y[silence_to_remove_start:]
    
    if silent_part_end_ms >= min_silence_length_ms:
        # Calculate the percentage to remove at the end
        silence_to_remove_end = int((len(y) - silent_part_end) * remove_percentage)
        y = y[:-silence_to_remove_end] if silence_to_remove_end > 0 else y

    return y

def process_wav(orgin_wav, orgin_sr, new_sr):
    
	udate_wav = remove_silence(orgin_wav, orgin_sr)
 	# Normalize the audio data
	if orgin_sr != new_sr:
		audio_data = librosa.resample(y=udate_wav, orig_sr=orgin_sr, target_sr=new_sr)
	else:
		audio_data = udate_wav    
	
	newlen = len(udate_wav)/float(new_sr)

	return udate_wav, newlen
