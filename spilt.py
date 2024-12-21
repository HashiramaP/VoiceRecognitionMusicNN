import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get the directory where the script is located - this is our project root
project_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
csv_file = os.path.join(project_dir, 'notesLabel.csv')
audio_dir = os.path.join(project_dir, 'notesRecording', 'splitFolder')

# Check if CSV file exists
if not os.path.exists(csv_file):
    print(f"Error: Could not find {csv_file}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project directory: {project_dir}")
    print("\nPlease ensure notesLabel.csv is in the same directory as this script.")
    exit(1)

# Read the CSV file
print(f"Reading CSV file from: {csv_file}")
df = pd.read_csv(csv_file)

# Add .wav extension if not present
df['filename'] = df['filename'].apply(lambda x: x if x.endswith('.wav') else x + '.wav')

# Add file_path column
df['file_path'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))

# Verify all files exist
missing_files = []
for file_path in df['file_path']:
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print("Warning: The following files are missing:")
    for file in missing_files:
        print(f"  - {file}")
    print("\nPlease check your data and try again.")
    exit(1)

# First split: 80% train+val, 20% test
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['note'])

# Second split: From the 80%, take 75% for train and 25% for validation (60% and 20% of total)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['note'])

# Define the column order
columns = ['filename', 'note', 'tempo', 'pitch', 'file_path']

# Save the splits to CSV files in the project directory
train_df[columns].to_csv(os.path.join(project_dir, 'train.csv'), index=False)
val_df[columns].to_csv(os.path.join(project_dir, 'val.csv'), index=False)
test_df[columns].to_csv(os.path.join(project_dir, 'test.csv'), index=False)

# Print summary
print(f"\nTotal samples: {len(df)}")
print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

print("\nFiles have been saved successfully!")