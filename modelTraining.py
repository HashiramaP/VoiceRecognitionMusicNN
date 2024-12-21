import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Shape: (13, time_steps)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    combined_features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)  # Shape: (39, time_steps)
    return torch.tensor(combined_features.T, dtype=torch.float32)  # Shape: (time_steps, 39)

import torch
import torch.nn as nn

class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Add dropout to the fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out

from torch.utils.data import DataLoader, Dataset

class AudioDataset(Dataset):
    def __init__(self, csv_file, label_encoder=None):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = label_encoder or LabelEncoder()
        self.data['note'] = self.label_encoder.fit_transform(self.data['note'])
        self.features = [extract_features(row['file_path']) for _, row in self.data.iterrows()]
        self.labels = torch.tensor(self.data['note'].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def pad_collate(batch):
    features, labels = zip(*batch)
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.stack(labels)
    return padded_features, labels


# Dataset and DataLoader
train_dataset = AudioDataset("train.csv")
val_dataset = AudioDataset("val.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 39  # 13 MFCCs + 13 Delta + 13 Delta-Delta
hidden_size = 128
num_layers = 2
num_classes = len(pd.read_csv("train.csv")['note'].unique())

model = AudioRNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate)

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

if __name__ == "__main__":
    for epoch in range(20):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on validation set
        val_accuracy = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{20}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Step the scheduler
        scheduler.step(val_accuracy)



    val_accuracy = evaluate(model, val_loader)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

