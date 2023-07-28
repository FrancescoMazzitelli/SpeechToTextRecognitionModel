import os
import librosa
import librosa.display
import numpy as np
import csv
import tensorflow as tf
import tensorflow_io as tfio
from PIL import Image
import matplotlib.pyplot as plt

# Decodifica del formato audio con restituzione di un tensore audio
def decode_audio(audio_binary):
  audio = tfio.audio.decode_mp3(input=audio_binary)
  return tf.squeeze(audio, axis=-1)

# Funzione che ritorna una forma d'onda, dopo averla decodificata
def get_waveform(file_path):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform

def audio_to_spectrogram(audio_tensor, sample_rate):
    # Trasformata di Fourier a breve termine (STFT) per ottenere lo spettrogramma
    spectrogram = librosa.stft(audio_tensor.numpy(), hop_length=512, n_fft=1024)

    # Converti lo spettrogramma in dB
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=np.max)

    return spectrogram_db

def extract_mfcc(audio_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """
    Estrae i Mel Frequency Cepstral Coefficients (MFCC) da un file audio.

    Argomenti:
    audio_path (str): Percorso del file audio.
    num_mfcc (int, opzionale): Numero di coefficienti MFCC da estrarre. Default: 13.
    n_fft (int, opzionale): Lunghezza della finestra per la Trasformata di Fourier a corto termine (STFT). Default: 2048.
    hop_length (int, opzionale): Numero di campioni tra le finestre STFT sovrapposte. Default: 512.

    Ritorna:
    numpy.ndarray: Un vettore contenente i coefficienti MFCC estratti.
    """
    
    # Carica il file audio utilizzando librosa
    y, sr = librosa.load(audio_path, sr=None)

    # Calcola gli MFCC utilizzando librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Restituisci i MFCC come matrice
    return mfccs
