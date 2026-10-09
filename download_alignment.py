"""Fetch the small multilingual Whisper model used for background alignment."""
from pathlib import Path

import whisper

if __name__ == "__main__":
    directory = Path(__file__).resolve().parent / "models" / "whisper"
    whisper.load_model("tiny", device="cpu", download_root=str(directory))
    print(directory / "tiny.pt")
