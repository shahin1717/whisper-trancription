# Whisper Audio Transcription

This is a Python-based project that uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe audio files. All transcriptions are automatically saved and appended into a single file in the `transcripts` folder.  

---

## Features

- Transcribe audio files (`.wav`, `.mp3`, etc.) using Whisper.
- Save all transcriptions to a central `transcripts/transcripts.txt` file.
- Timestamp each transcription for easy reference.
- Simple command-line interface (no GUI needed).

---

## Requirements

- Python 3.10+  
- [Whisper](https://github.com/openai/whisper) (`pip install git+https://github.com/openai/whisper.git`)  
Optional if you want to record from the microphone:

- `sounddevice`  
- `numpy`  
---

## Usage

1. Place your audio file inside the `recordings` folder.
2. Run the transcription script from terminal:

```bash
python model.py recordings/audio_file.wav
