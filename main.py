import torch
import torchaudio
import speech_recognition as sr
from modelTraining import AudioRNN, extract_features  # Ensure the model and feature extraction function are imported
import pandas as pd

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 39  # 13 MFCCs + 13 Delta + 13 Delta-Delta
hidden_size = 128
num_layers = 2
num_classes = len(pd.read_csv("train.csv")['note'].unique())  # Ensure this matches the training data
model = AudioRNN(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

# Label mapping (update based on your training labels)
LABELS = {0: "Label1", 1: "Label2", 2: "Label3"}  # Replace with actual labels

def preprocess_audio(file_path):
    features = extract_features(file_path)  # Extract MFCC + Delta features
    return features.unsqueeze(0)  # Add batch dimension

def predict(file_path):
    features = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    return LABELS.get(predicted.item(), "Unknown")

# Set up microphone input using SpeechRecognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Listening loop
print("Listening... Press Ctrl+C to stop.")
with microphone as source:
    while True:
        try:
            print("Say something...")
            audio = sr.Recognizer().listen(source)  # Capture audio
            # Save audio to a temporary file
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            # Extract audio features
            audio_features = extract_features("temp_audio.wav")
            print(f"Audio features: {audio_features.shape}")  # Log audio features shape

            # Add batch dimension to features to match LSTM input shape
            audio_features = audio_features.unsqueeze(0).to(device)  # Now shape: (1, time_steps, 39)
            
            # Make prediction with the model
            output = model(audio_features)  # Adjust based on your model input/output
            print(f"Model output: {output}")  # Log the raw model output

            # Process model output (e.g., classification)
            _, predicted_class = torch.max(output, 1)
            print(f"Predicted: {predicted_class.item()}")  # Print predicted class

        except KeyboardInterrupt:
            print("Exiting...")
            break