from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import os
import re
import json
from typing import List
from datetime import datetime 

INTENT_PHRASES_PATH = "intent_phrases.json"  # Changed to .json

app = FastAPI()

# Allow requests from all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load intent phrases from JSON
# ---------------------------

AUDIO_STORAGE_DIR = "uploaded_audio"
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)


def load_intent_phrases(filepath: str = INTENT_PHRASES_PATH) -> List[str]:
    """
    Load phrases from JSON file, extract values from search_keywords array,
    and sort longest first.
    """
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract the 'value' field from each object in search_keywords array
        phrases = []
        if "search_keywords" in data and isinstance(data["search_keywords"], list):
            phrases = [item["value"].strip() for item in data["search_keywords"] 
                      if item.get("value") and item["value"].strip()]
        
        # Sort by length (longest first) for proper removal order
        phrases.sort(key=len, reverse=True)
        return phrases
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error loading intent phrases from JSON: {e}")
        return []


# ---------------------------
# Remove intent/filler phrases (same as before)
# ---------------------------
def clean_text_remove_intent_phrases(text: str, intent_phrases: List[str]) -> str:
    """
    Remove known intent/filler phrases (case-insensitive) and return cleaned text.
    """
    if not text:
        return ""

    cleaned = text
    for phrase in intent_phrases:
        if not phrase:
            continue
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
        cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------
# Transcription Endpoint (Raw text - WITH intent phrases)
# ---------------------------
@app.post("/trans")
async def transcribe_both(file: UploadFile = File(...)):
    """
    Returns both raw and cleaned transcription in a single response.
    """
    recognizer = sr.Recognizer()
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
    wav_path = None

    try:
        # Save uploaded file temporarily
        file_content = await file.read()
        temp_in.write(file_content)
        temp_in.flush()
        temp_in.close()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        saved_path = os.path.join(AUDIO_STORAGE_DIR, f"{timestamp}_{file.filename}")
        with open(saved_path, "wb") as f:
            f.write(file_content)

        # Convert to WAV using pydub
        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name + ".wav"
        audio.export(wav_path, format="wav")

        # Transcribe audio
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            raw_text = recognizer.recognize_google(audio_data)

            # Load phrases and clean text
            intent_phrases = load_intent_phrases()
            cleaned_text = clean_text_remove_intent_phrases(raw_text, intent_phrases)

            # Print both versions to terminal
            print(f"\n Raw transcription: {raw_text}")
            print(f" Cleaned query: {cleaned_text}\n")

            # Return both texts
            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned_text
            }

    except sr.UnknownValueError:
        print("Could not understand the audio")
        return {"raw_text": "", "cleaned_text": ""}
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return {"raw_text": "", "cleaned_text": ""}
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"raw_text": "", "cleaned_text": ""}
    finally:
        for path in [temp_in.name, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

# uvicorn voice:app --host 10.0.17.101 --port 8001 --reload