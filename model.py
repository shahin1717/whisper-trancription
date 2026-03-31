import whisper
import sounddevice as sd
import numpy as np
import datetime
import os
import sys

# ------------------ CONFIG ------------------
DURATION = 10       # seconds per recording
FS = 16000          # sampling rate
MODEL_NAME = "small" # "tiny", "base", "small", "medium", "large"
SAVE_FOLDER = "transcripts"

os.makedirs(SAVE_FOLDER, exist_ok=True)


#!------------------ RECORDING TRANSCRIPTION FUNCTION ------------------

# ------------------ LOAD MODEL ------------------
print(f"Loading Whisper model '{MODEL_NAME}' ...")
model = whisper.load_model(MODEL_NAME)

# ------------------ RECORDING FUNCTION ------------------
def record_audio(duration=DURATION, fs=FS):
    print(f"\nRecording for {duration} seconds. Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete!")
    return audio.flatten().astype(np.float32)

# ------------------ TRANSCRIPTION FUNCTION ------------------
def transcribe_audio(audio):
    result = model.transcribe(audio, fp16=False)
    return result["text"]




def main1():
    while True:
        audio = record_audio()
        text = transcribe_audio(audio)
        print("\n--- TRANSCRIPTION ---")
        print(text)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_FOLDER, f"transcript_{timestamp}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved transcription: {filename}")
        
        cont = input("\nRecord again? (y/n): ").strip().lower()
        if cont != "y":
            break



#!------------------ FILE TRANSCRIPTION FUNCTION ------------------

def transcribe_file(file_path):
    if not os.path.isfile(f"recordings/{file_path}"):
        print(f"Error: File '{file_path}' does not exist.")
        return

    print(f"Loading Whisper model 'small' ...")
    model = whisper.load_model("small")

    print(f"Transcribing '{file_path}' ...")
    result = model.transcribe(f"recordings/{file_path}")
    print("\n--- Transcription ---\n")
    print(result["text"])
    print("\n--------------------\n")
    # Append transcription to the single file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SAVE_FOLDER + "/transcriptions.txt", "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] {file_path}\n{result['text']}\n{'-'*50}\n")
    print(f"Appended transcription to '{SAVE_FOLDER}/transcriptions.txt'")

def main2():
    if len(sys.argv) < 2:
        print("Usage: python model.py <audio_file>")
        return

    audio_file = sys.argv[1]
    transcribe_file(audio_file)
    
if __name__ == "__main__":
    main2()