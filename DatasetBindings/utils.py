import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
#import pygame
import csv

#Funzione per il caricamento delle trascrizioni degli audio
def find_transcripts(path, operation):
    if os.path.exists(path):
        full_path = os.path.join(path, operation)
        for filename in os.listdir(full_path):
            if filename.endswith("transcripts.txt"):
                file_path = os.path.join(full_path, filename)
                break

        if file_path is not None:
            try:
                with open(file_path, 'r') as file:
                    transcripts = file.read()
                return transcripts
            except FileNotFoundError:
                return None
        else:
            return None
        

def bind_audio_transcripts(path, operation):
    full_path = os.path.join(path, operation)
    bind_audio_transcripts = {}
    for row in find_transcripts(path, operation).splitlines():
        
        row_split = row.split('	')
        audio_code = row_split[0]
        audio_transcript = row_split[1]
        
        audio_code_split = audio_code.split('_')
        first_folder = audio_code_split[0]
        second_folder = audio_code_split[1]

        audio_path = os.path.join(full_path, "audio", first_folder, second_folder)
        for audio in os.listdir(audio_path):
            if audio.endswith(audio_code + ".opus"):
                full_audio_path = os.path.join(audio_path, audio_code + ".opus")
                #play_audio(full_audio_path)
                bind_audio_transcripts[full_audio_path] = audio_transcript
    return bind_audio_transcripts
        

# Funzione per il preprocessing delle tracce audio
def preprocess_audio(audio_path, sr=16000):
    # Carica l'audio con librosa
    audio_data, _ = librosa.load(audio_path, sr=sr)

    # Rimuovi il rumore di fondo
    audio_data = librosa.effects.trim(audio_data, top_db=20)[0]

    # Normalizza il volume
    audio_data /= max(abs(audio_data))

    return audio_data


def save_dict_to_csv(dict_data, file_path):
    with open(file_path, 'w+', newline='') as file:
        writer = csv.writer(file)

        # Scrive l'intestazione delle colonne (le chiavi del dizionario)
        writer.writerow(["Traccia", "trascrizione"])

        # Scrive i dati delle righe (i valori del dizionario)
        for key, value in dict_data.items():
            writer.writerow([key, value])

"""
#Solo a scopo di debug
def play_audio(file_path):
    # Inizializza pygame
    pygame.init()

    # Imposta il volume (opzionale)
    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.5)  # Imposta il volume al 50%

    # Carica il file audio
    pygame.mixer.music.load(file_path)

    # Avvia la riproduzione
    pygame.mixer.music.play()

    # Aspetta fino a quando la riproduzione non è completata
    while pygame.mixer.music.get_busy():
        continue

    # Chiudi pygame
    pygame.quit()
"""
