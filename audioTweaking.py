import os
import random
import librosa
import soundfile as sf
import pandas as pd
import scipy.signal

# Define the directories
audio_dir = 'notesRecording/splitFolder/'
csv_file = 'notesLabel.csv'

# Load the existing CSV
df = pd.read_csv(csv_file)

# Function to apply pitch shift using librosa
def apply_pitch_shift(audio_file, semitones):
    y, sr = librosa.load(audio_file)
    # Fix: Use keyword argument for sr parameter
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)
    new_file = audio_file.replace('.wav', f'_pitch_shifted_{semitones}.wav')
    sf.write(new_file, y_shifted, sr)
    return new_file

# Function to apply time-stretching using librosa
def time_stretch(audio_file, stretch_factor):
    y, sr = librosa.load(audio_file)
    # Calculate the new length after stretching
    new_length = int(len(y) * stretch_factor)
    y_stretched = scipy.signal.resample(y, new_length)
    new_file = audio_file.replace('.wav', f'_time_stretched_{stretch_factor}.wav')
    sf.write(new_file, y_stretched, sr)
    return new_file

# List of possible tempo and pitch tweaks
tempo_changes = ['normal', 'fast', 'slow']
pitch_changes = ['normal', 'low', 'high']

# Prepare a list to store new rows
new_rows = []

# Loop through each file in the folder
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        
        # Randomly tweak tempo and pitch for this audio file
        tempo = random.choice(tempo_changes)
        pitch = random.choice(pitch_changes)
        
        # Create a copy of the original file for processing
        current_file = file_path
        
        # Apply tempo change
        if tempo == 'fast':
            current_file = time_stretch(current_file, 1.2)  # Stretch 120% faster
        elif tempo == 'slow':
            current_file = time_stretch(current_file, 0.8)  # Stretch 80% slower
        
        # Apply pitch shift if necessary
        if pitch == 'low':
            current_file = apply_pitch_shift(current_file, -2)  # Lower pitch by 2 semitones
        elif pitch == 'high':
            current_file = apply_pitch_shift(current_file, 2)  # Raise pitch by 2 semitones
        
        # Extract note from the original filename
        note = filename.split('_')[0]
        
        # Create the new row for the CSV
        new_row = {
            'filename': os.path.basename(current_file),  # Store only the filename, not full path
            'note': note,
            'tempo': tempo,
            'pitch': pitch
        }
        
        # Append the new row to the list
        new_rows.append(new_row)

# Convert the list of new rows to a DataFrame
new_df = pd.DataFrame(new_rows)

# Concatenate the new DataFrame with the original one
df = pd.concat([df, new_df], ignore_index=True)

# Save the updated CSV
df.to_csv(csv_file, index=False)

print(f"Updated CSV: {csv_file}")