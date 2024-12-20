from pydub import AudioSegment
import os

# Load M4A audio file
audio = AudioSegment.from_file("notesRecording/Do.m4a", format="m4a")

# Define the length of each chunk in milliseconds (5 seconds)
chunk_length_ms = 5 * 1000  # 5 seconds = 5000 milliseconds

# Create the output directory if it doesn't exist
output_dir = "notesRecording/splitFolder/DoSplit"
os.makedirs(output_dir, exist_ok=True)

# Split the audio into 5-second chunks and save them
for i in range(0, len(audio), chunk_length_ms):
    chunk = audio[i:i + chunk_length_ms]
    chunk.export(f"{output_dir}/do_{i // chunk_length_ms}.wav", format="wav")
