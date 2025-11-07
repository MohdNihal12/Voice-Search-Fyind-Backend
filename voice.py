from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import os
import re
from typing import List
from jiwer import wer, cer

INTENT_PHRASES_PATH = "intent_phrases.txt"

app = FastAPI()

# Allow requests from all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load intent phrases
# ---------------------------
def load_intent_phrases(filepath: str = INTENT_PHRASES_PATH) -> List[str]:
    """
    Load phrases from file, remove empty lines, and sort longest first.
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    phrases.sort(key=len, reverse=True)
    return phrases


# ---------------------------
# Remove intent/filler phrases
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
# Transcription Endpoint
# ---------------------------
@app.post("/trans")
async def transcribe(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
    wav_path = None

    try:
        # Save uploaded file temporarily
        temp_in.write(await file.read())
        temp_in.flush()
        temp_in.close()

        # Convert to WAV using pydub (requires ffmpeg)
        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name + ".wav"
        # print(wav_path)
        audio.export(wav_path, format="wav")

        # Transcribe audio
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

            # Load phrases and clean text
            intent_phrases = load_intent_phrases()
            cleaned = clean_text_remove_intent_phrases(text, intent_phrases)

            # Print both versions to terminal
            print(f"\n Raw transcription: {text}")
            print(f" Cleaned query: {cleaned}\n")

            # Return only cleaned text to frontend
            return {"text": cleaned}

    except sr.UnknownValueError:
        print("Could not understand the audio")
        return {"text": ""}
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return {"text": ""}
    except Exception as e:
        print(f" Error during transcription: {e}")
        return {"text": ""}
    finally:
        # Always clean up temporary files
        for path in [temp_in.name, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


#trascription evaluation endpoint
@app.post("/evaluate")
async def evaluate_transcription(file: UploadFile = File(...), ground_truth: str = Form(...)):
    recognizer = sr.Recognizer()
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
    wav_path = None

    try:
        temp_in.write(await file.read())
        temp_in.flush()
        temp_in.close()

        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name + ".wav"
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            predicted_text = recognizer.recognize_google(audio_data)

        # Clean the text
        intent_phrases = load_intent_phrases()
        cleaned_pred = clean_text_remove_intent_phrases(predicted_text, intent_phrases)
        cleaned_truth = clean_text_remove_intent_phrases(ground_truth, intent_phrases)

        # Compute metrics
        word_error_rate = wer(cleaned_truth.lower(), cleaned_pred.lower())
        char_error_rate = cer(cleaned_truth.lower(), cleaned_pred.lower())
        accuracy = (1 - char_error_rate) * 100

        report = {
            "ground_truth": cleaned_truth,
            "predicted_text": cleaned_pred,
            "word_error_rate": round(word_error_rate, 3),
            "char_error_rate": round(char_error_rate, 3),
            "accuracy_percent": round(accuracy, 2)
        }

        print("\n--- Accuracy Report ---")
        for k, v in report.items():
            print(f"{k}: {v}")
        print("-----------------------\n")

        return report

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"error": str(e)}

    finally:
        for path in [temp_in.name, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)


################################################
# uvicorn voice:app --host 10.0.17.101  --port 8001 --reload