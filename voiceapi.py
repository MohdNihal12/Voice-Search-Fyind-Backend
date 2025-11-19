from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import os
import re
import json
from typing import List
from datetime import datetime 
import uuid
from pydub import AudioSegment

INTENT_PHRASES_PATH = "intent_phrases.json"
TRANSCRIPTIONS_PATH = "transcriptions.json"

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


def save_intent_phrases(intent_phrases: List[dict], filepath: str = INTENT_PHRASES_PATH) -> bool:
    """
    Save intent phrases to JSON file.
    Returns True if successful, False otherwise.
    """
    try:
        # Create the data structure
        data = {
            "search_keywords": intent_phrases
        }
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving intent phrases: {e}")
        return False


def get_next_id(intent_phrases: List[dict]) -> int:
    """
    Get the next available ID for new intent phrases.
    """
    if not intent_phrases:
        return 1
    
    max_id = max(phrase.get("id", 0) for phrase in intent_phrases)
    return max_id + 1


def save_transcription(raw_text: str, cleaned_text: str, audio_filename: str, audio_file_path: str) -> dict:
    """
    Save transcription data to JSON file with timestamp and unique ID.
    Returns the saved transcription object.
    """
    try:
        # Create transcription object
        transcription_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        transcription_data = {
            "id": transcription_id,
            "timestamp": timestamp,
            "audio_file_path": audio_file_path,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text
        }
        
        # Load existing transcriptions or create new file
        transcriptions = []
        if os.path.exists(TRANSCRIPTIONS_PATH):
            try:
                with open(TRANSCRIPTIONS_PATH, "r", encoding="utf-8") as f:
                    transcriptions = json.load(f)
            except (json.JSONDecodeError, Exception):
                transcriptions = []
        
        # Add new transcription
        transcriptions.append(transcription_data)
        
        # Save back to file - FIXED: removed duplicate ensure_ascii=False
        with open(TRANSCRIPTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)
        
        print(f"Transcription saved with ID: {transcription_id}")
        return transcription_data
        
    except Exception as e:
        print(f"Error saving transcription: {e}")
        return None


# ---------------------------
# Intent Phrases Management Endpoints
# ---------------------------

@app.get("/intent-phrases")
async def get_intent_phrases():
    """
    Get all intent phrases.
    """
    try:
        if not os.path.exists(INTENT_PHRASES_PATH):
            return {"search_keywords": []}
        
        with open(INTENT_PHRASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading intent phrases: {str(e)}")


@app.post("/intent-phrases")
async def add_intent_phrase(phrase: str = Form(...)):
    """
    Add a new intent phrase.
    """
    try:
        phrase = phrase.strip()
        if not phrase:
            raise HTTPException(status_code=400, detail="Phrase cannot be empty")
        
        # Load existing phrases
        if os.path.exists(INTENT_PHRASES_PATH):
            with open(INTENT_PHRASES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"search_keywords": []}
        
        # Check if phrase already exists
        existing_phrases = [p["value"].lower() for p in data["search_keywords"]]
        if phrase.lower() in existing_phrases:
            raise HTTPException(status_code=400, detail="Phrase already exists")
        
        # Add new phrase
        new_phrase = {
            "id": get_next_id(data["search_keywords"]),
            "value": phrase
        }
        data["search_keywords"].append(new_phrase)
        
        # Save back to file
        if save_intent_phrases(data["search_keywords"]):
            return {"message": "Phrase added successfully", "phrase": new_phrase}
        else:
            raise HTTPException(status_code=500, detail="Failed to save phrase")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding intent phrase: {str(e)}")


@app.delete("/intent-phrases/{phrase_id}")
async def delete_intent_phrase(phrase_id: int):
    """
    Delete an intent phrase by ID.
    """
    try:
        if not os.path.exists(INTENT_PHRASES_PATH):
            raise HTTPException(status_code=404, detail="No intent phrases found")
        
        with open(INTENT_PHRASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Find and remove the phrase
        original_count = len(data["search_keywords"])
        data["search_keywords"] = [p for p in data["search_keywords"] if p.get("id") != phrase_id]
        
        if len(data["search_keywords"]) == original_count:
            raise HTTPException(status_code=404, detail="Phrase not found")
        
        # Save back to file
        if save_intent_phrases(data["search_keywords"]):
            return {"message": "Phrase deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete phrase")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting intent phrase: {str(e)}")


@app.put("/intent-phrases/{phrase_id}")
async def update_intent_phrase(phrase_id: int, new_phrase: str = Form(...)):
    """
    Update an existing intent phrase.
    """
    try:
        new_phrase = new_phrase.strip()
        if not new_phrase:
            raise HTTPException(status_code=400, detail="Phrase cannot be empty")
        
        if not os.path.exists(INTENT_PHRASES_PATH):
            raise HTTPException(status_code=404, detail="No intent phrases found")
        
        with open(INTENT_PHRASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Find and update the phrase
        phrase_found = False
        for phrase in data["search_keywords"]:
            if phrase.get("id") == phrase_id:
                phrase["value"] = new_phrase
                phrase_found = True
                break
        
        if not phrase_found:
            raise HTTPException(status_code=404, detail="Phrase not found")
        
        # Save back to file
        if save_intent_phrases(data["search_keywords"]):
            return {"message": "Phrase updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update phrase")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating intent phrase: {str(e)}")


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
    Returns both raw and cleaned transcription in a single response and saves to JSON.
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

        # Save original uploaded audio file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        saved_audio_path = os.path.join(AUDIO_STORAGE_DIR, f"{timestamp}_{file.filename}")
        with open(saved_audio_path, "wb") as f:
            f.write(file_content)

        # ----------------------------
        # Convert to high-quality WAV 
        # ----------------------------
        audio = AudioSegment.from_file(temp_in.name)
        if len(audio) < 500:  
            padding = AudioSegment.silent(duration=300)
            audio = padding + audio + padding

        # Enhance audio clarity
        audio = audio.normalize()
        audio = audio.set_channels(1)         # mono
        audio = audio.set_frame_rate(44100)   # 44.1 kHz

        wav_path = temp_in.name + ".wav"
        audio.export(
            wav_path,
            format="wav",
            parameters=["-ac", "1", "-ar", "44100"]
        )

        # ----------------------------
        # Transcribe audio
        # ----------------------------
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            raw_text = recognizer.recognize_google(audio_data)

        # ----------------------------
        # Clean text using intent phrases
        # ----------------------------
        intent_phrases = load_intent_phrases()
        cleaned_text = clean_text_remove_intent_phrases(raw_text, intent_phrases)

        # Print to console
        print(f"\n Raw transcription: {raw_text}")
        print(f" Cleaned query: {cleaned_text}\n")

        # ----------------------------
        # Save transcription to JSON
        # ----------------------------
        transcription_data = save_transcription(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            audio_filename=file.filename,
            audio_file_path=saved_audio_path
        )

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "transcription_id": transcription_data["id"] if transcription_data else None,
            "message": "Transcription saved successfully" if transcription_data else "Transcription completed but failed to save"
        }

    except sr.UnknownValueError:
        return {"raw_text": "", "cleaned_text": "", "transcription_id": None, "message": "Could not understand the audio"}

    except sr.RequestError as e:
        return {"raw_text": "", "cleaned_text": "", "transcription_id": None, "message": f"Speech recognition service error: {e}"}

    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"raw_text": "", "cleaned_text": "", "transcription_id": None, "message": f"Error during transcription: {e}"}

    finally:
        # cleanup temp files
        for path in [temp_in.name, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


# uvicorn voice:app --host 10.0.17.101 --port 8004 --reload