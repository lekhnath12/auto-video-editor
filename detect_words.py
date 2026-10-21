import os
import logging
from typing import List, Tuple
import json
import time
import re

from google import genai
from google.genai import types
from google.genai.errors import APIError
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def repair_and_parse_json(json_str: str) -> dict:
    """Robustly parse JSON, attempting fixes for truncated outputs or trailing commas."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON parse failed ({e}). Attempting auto-repair...")
        
        # Clean markdown code blocks if present
        cleaned = re.sub(r"^```json\s*", "", json_str.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # If truncated, try closing open brackets/braces
        fixed = cleaned
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        
        # Remove trailing incomplete key/value or comma
        fixed = re.sub(r',\s*("[^"]*")?\s*$', '', fixed)
        
        fixed += '}' * max(0, open_braces)
        fixed += ']' * max(0, open_brackets)
        
        try:
            parsed = json.loads(fixed)
            logging.info("Successfully repaired and parsed truncated JSON response.")
            return parsed
        except json.JSONDecodeError as final_err:
            logging.error(f"Could not repair JSON: {final_err}")
            raise final_err


def generate_fine_grained_transcription(audio_path: str) -> Tuple[List[dict], str]:
    """
    Transcribes audio using Gemini to produce a precise, fine-grained sequence 
    containing every word, silence block, or unpredicted word/sound with start and end timestamps.
    """
    raw_transcript_path = audio_path.rsplit(".", 1)[0] + "_transcript.json"
        
    if os.path.exists(raw_transcript_path):
        logging.info(f"Found existing fine-grained transcription cache: {raw_transcript_path}. Skipping Gemini inference.")
        try:
            with open(raw_transcript_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            timed_elements = result_data.get("timed_elements", [])
            transcript_text = result_data.get("transcript", "")
            logging.info(f"Loaded {len(timed_elements)} elements from cache.")
            return timed_elements, transcript_text
        except Exception as cache_err:
            logging.warning(f"Failed to read cache file ({cache_err}), proceeding with Gemini API call...")

    logging.info("Initializing Google GenAI client for fine-grained transcription...")
    client = genai.Client()
        
    logging.info(f"Uploading audio file to Gemini: {audio_path}")
    audio_file = client.files.upload(file=audio_path)
    
    try:
        prompt = """
        You are an expert audio transcriber and phonetic analyst. The audio is spoken in Nepali (and may include English words or background noises).
        
        Provide a sequential, chronological timeline breakdown of the entire audio stream (`timed_elements`) into discrete words, silence durations, or unpredicted sounds/fillers.
        For each element, specify:
        - "word": the text representation of the word, silence descriptor (e.g., "[silence]"), or unpredicted sound/word.
        - "start": start time in seconds (floating point).
        - "end": end time in seconds (floating point).
        - "type": classification of the item, such as "word", "silence", or "filler".

        For example, 
        {
            "word": "गो",
            "start": 0.0,
            "end": 0.44,
            "type": "word"
        },
           
        Return your output strictly as a JSON object with fields with a list of objects, each with keys "word", "start", "end", and "type".
        """
        
        response = None
        attempt = 1
        
        while True:
            try:
                logging.info(f"Sending request to Gemini (gemini-3.8-flash) [Attempt {attempt}]...")
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[audio_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=65536, # Increased token capacity for long audio files
                    ),
                )
                break
                
            except APIError as e:
                if e.code == 503 or "503" in str(e) or "UNAVAILABLE" in str(e):
                    logging.warning(f"Gemini API returned 503 Service Unavailable (Attempt {attempt}). Waiting for 1 minute before retrying...")
                    time.sleep(60)
                    attempt += 1
                else:
                    raise e
            except Exception as e:
                err_str = str(e)
                if any(x in err_str for x in ["503", "Service Unavailable", "UNAVAILABLE", "disconnected", "timed out", "Connection aborted"]):
                    logging.warning(f"Encountered connection interruption (Attempt {attempt}): {e}. Waiting for 1 minute...")
                    time.sleep(60)
                    attempt += 1
                else:
                    raise e

        # Safely parse response with auto-repair fallback
        result_data = repair_and_parse_json(response.text)

        timed_elements = result_data.get("timed_elements", [])
        transcript_text = result_data.get("transcript", "")
        
        logging.info(f"Saving fine-grained transcript data to: {raw_transcript_path}")
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
            
        return timed_elements, transcript_text
        
    except Exception as e:
        logging.error(f"Error during Gemini fine-grained transcription: {e}")
        return [], ""
        
    finally:
        try:
            client.files.delete(name=audio_file.name)
            logging.info("Cleaned up audio file from Gemini storage.")
        except Exception as cleanup_err:
            logging.warning(f"Failed to delete remote audio file: {cleanup_err}")


if __name__ == "__main__":
    audio_path = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.47.21 Lekh Poudel's Zoom Meeting\audio1103826994.m4a"
    timed_elements, transcript = generate_fine_grained_transcription(audio_path=audio_path)
    print(f"Successfully transcribed {len(timed_elements)} elements.")