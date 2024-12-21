import torch
import torchaudio
import speech_recognition as sr
from torch.nn.utils.rnn import pad_sequence
from modelTraining import AudioRNN, extract_features  # Ensure the model and feature extraction function are imported

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 39  # 13 MFCCs + 13 Delta + 13 Delta-Delta
hidden_size = 128
num_layers = 2
num_classes = 10  # Replace with the actual number of classes from your training

model = AudioRNN(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
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

print("Listening... Press Ctrl+C to stop.")

try:
    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Say something...")
            audio = recognizer.listen(source)

        # Save audio to a temporary file
        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())

        # Predict the spoken word
        prediction = predict("temp.wav")
        print(f"Predicted: {prediction}")

except KeyboardInterrupt:
    print("\nExiting...")
