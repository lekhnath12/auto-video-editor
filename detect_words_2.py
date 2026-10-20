import os
import logging
from typing import List, Tuple
import json
import time
import re
import subprocess

from google import genai
from google.genai import types
from google.genai.errors import APIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def repair_and_parse_json(json_str: str) -> dict:
    """Robustly parse JSON, attempting fixes for truncated outputs or trailing commas."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON parse failed ({e}). Attempting auto-repair...")
        
        cleaned = re.sub(r"^```json\s*", "", json_str.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        fixed = cleaned
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        
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


def get_audio_duration(audio_path: str) -> float:
    """Get exact audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())


def split_audio_into_chunks(audio_path: str, chunk_duration_sec: int = 300) -> List[Tuple[str, float]]:
    """Splits audio into 5-minute chunks using FFmpeg and returns a list of (chunk_path, start_offset_seconds)."""
    total_duration = get_audio_duration(audio_path)
    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    chunks = []
    start_time = 0.0
    chunk_idx = 0
    
    while start_time < total_duration:
        chunk_filename = f"{base_name}_chunk_{chunk_idx:03d}.m4a"
        chunk_path = os.path.join(base_dir, chunk_filename)
        
        # FFmpeg segment extraction command
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", audio_path,
            "-t", str(chunk_duration_sec), "-c", "copy", chunk_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        chunks.append((chunk_path, start_time))
        start_time += chunk_duration_sec
        chunk_idx += 1
        
    logging.info(f"Split audio into {len(chunks)} chunks of {chunk_duration_sec}s each.")
    return chunks


def transcribe_single_chunk(client: genai.Client, chunk_path: str, offset: float) -> Tuple[List[dict], str]:
    """Uploads and transcribes a single audio chunk via Gemini, shifting timestamps by offset."""
    logging.info(f"Uploading chunk to Gemini: {os.path.basename(chunk_path)} (Offset: {offset}s)")
    audio_file = client.files.upload(file=chunk_path)
    
    try:
        prompt = """
        You are an expert audio transcriber and phonetic analyst. The audio is spoken in Nepali (and may include English words or background noises).
        
        1. Provide the full verbatim transcript string of this audio segment in the "transcript" field.
        2. Provide a sequential timeline breakdown of this audio segment (`timed_elements`) into discrete words, silence durations, or unpredicted sounds/fillers.
        For each element, specify:
        - "word": the text representation of the word, silence descriptor (e.g., "[silence]"), or unpredicted sound/word.
        - "start": start time in seconds relative to the beginning of this audio chunk (floating point).
        - "end": end time in seconds relative to the beginning of this audio chunk (floating point).
        - "type": classification of the item, such as "word", "silence", or "filler".
           
        Return your output strictly as a JSON object with two fields:
        - "transcript": string containing the transcript for this chunk.
        - "timed_elements": a list of objects, each with keys "word", "start", "end", and "type".
        """
        
        attempt = 1
        while True:
            try:
                logging.info(f"Sending chunk request to Gemini (gemini-3.8-flash) [Attempt {attempt}]...")
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[audio_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=65536,
                    ),
                )
                break
            except APIError as e:
                if e.code == 503 or "UNAVAILABLE" in str(e):
                    logging.warning(f"Service unavailable. Waiting 60s before retry...")
                    time.sleep(60)
                    attempt += 1
                else:
                    raise e

        result_data = repair_and_parse_json(response.text)
        chunk_timed_elements = result_data.get("timed_elements", [])
        chunk_transcript = result_data.get("transcript", "")
        
        # Shift all timestamps by the global chunk offset
        shifted_elements = []
        for el in chunk_timed_elements:
            shifted_elements.append({
                "word": el.get("word", ""),
                "start": round(el.get("start", 0.0) + offset, 2),
                "end": round(el.get("end", 0.0) + offset, 2),
                "type": el.get("type", "word")
            })
            
        return shifted_elements, chunk_transcript

    finally:
        try:
            client.files.delete(name=audio_file.name)
            logging.info(f"Cleaned up chunk from Gemini storage: {os.path.basename(chunk_path)}")
        except Exception:
            pass


def generate_fine_grained_transcription(audio_path: str) -> Tuple[List[dict], str]:
    raw_transcript_path = audio_path.rsplit(".", 1)[0] + "_transcript.json"
        
    if os.path.exists(raw_transcript_path):
        logging.info(f"Found existing fine-grained transcription cache: {raw_transcript_path}. Skipping Gemini inference.")
        try:
            with open(raw_transcript_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            return result_data.get("timed_elements", []), result_data.get("transcript", "")
        except Exception as cache_err:
            logging.warning(f"Failed to read cache file ({cache_err}), proceeding with chunked processing...")

    logging.info("Initializing Google GenAI client for chunked transcription...")
    client = genai.Client()
    
    # 1. Split audio into 5-minute (300 seconds) chunks
    chunks = split_audio_into_chunks(audio_path, chunk_duration_sec=300)
    
    all_timed_elements = []
    all_transcripts = []
    
    try:
        # 2. Transcribe each chunk sequentially and combine
        for chunk_path, offset in chunks:
            timed_elements, chunk_text = transcribe_single_chunk(client, chunk_path, offset)
            all_timed_elements.extend(timed_elements)
            if chunk_text:
                all_transcripts.append(chunk_text)
                
        full_transcript = " ".join(all_transcripts)
        
        final_result = {
            "transcript": full_transcript,
            "timed_elements": all_timed_elements
        }
        
        logging.info(f"Saving combined fine-grained transcript data to: {raw_transcript_path}")
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
            
        return all_timed_elements, full_transcript
        
    except Exception as e:
        logging.error(f"Error during chunked transcription workflow: {e}")
        return [], ""
        
    finally:
        # 3. Clean up temporary chunk audio files from local disk
        for chunk_path, _ in chunks:
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass


if __name__ == "__main__":
    audio_path = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.47.21 Lekh Poudel's Zoom Meeting\audio1103826994.m4a"
    timed_elements, transcript = generate_fine_grained_transcription(audio_path=audio_path)
    print(f"Successfully transcribed {len(timed_elements)} elements across all chunks.")