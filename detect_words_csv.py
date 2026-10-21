import os
import logging
from typing import List, Tuple
import json
import csv
import io
import time
import re
import subprocess
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_csv_response(csv_str: str) -> List[dict]:
    """Parse pipe-separated or comma-separated CSV response text into timed elements."""
    timed_elements = []
    
    # Strip markdown code blocks if present
    cleaned = re.sub(r"^```(?:csv)?\s*", "", csv_str.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    
    # Auto-detect delimiter based on prompt format (pipe '|' or comma ',')
    delimiter = '|' if '|' in cleaned else ','
    
    f = io.StringIO(cleaned)
    reader = csv.reader(f, delimiter=delimiter)
    for row in reader:
        if not row or len(row) < 4:
            continue
        # Skip header rows if present
        if row[0].strip().lower() in ("word", "text") and row[1].strip().lower() in ("start", "s"):
            continue
        try:
            word = row[0].strip()
            start = float(row[1].strip())
            end = float(row[2].strip())
            elem_type = row[3].strip()
            timed_elements.append({
                "word": word,
                "start": start,
                "end": end,
                "type": elem_type
            })
        except ValueError as e:
            logging.warning(f"Skipping malformed CSV row {row}: {e}")
            
    return timed_elements


def get_audio_duration(audio_path: str) -> float:
    """Get exact audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())


def split_audio_into_chunks(audio_path: str, chunk_duration_sec: int = 300, overlap_sec: int = 30, debug_dir: str = None) -> List[Tuple[str, float]]:
    """Splits audio into overlapping chunks using FFmpeg and returns a list of (chunk_path, start_offset_seconds)."""
    total_duration = get_audio_duration(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    chunks = []
    start_time = 0.0
    chunk_idx = 0
    
    while start_time < total_duration:
        current_chunk_duration = chunk_duration_sec if (start_time + chunk_duration_sec) < total_duration else (total_duration - start_time)
        
        chunk_filename = f"{base_name}_chunk_{chunk_idx:03d}.wav"
        chunk_path = os.path.join(debug_dir, chunk_filename) if debug_dir else chunk_filename
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", audio_path,
            "-t", str(current_chunk_duration), 
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
            chunk_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        chunks.append((chunk_path, start_time))
        
        step_size = chunk_duration_sec - overlap_sec
        start_time += step_size
        chunk_idx += 1
        
    logging.info(f"Split audio into {len(chunks)} chunks of {chunk_duration_sec}s each with {overlap_sec}s overlap.")
    return chunks


def transcribe_single_chunk(client: genai.Client, chunk_path: str, offset: float, chunk_idx: int, debug_dir: str = None) -> Tuple[List[dict], str]:
    """Uploads and transcribes a single audio chunk via Gemini, shifting timestamps by offset."""
    chunk_duration = get_audio_duration(chunk_path)
    
    # Check if cached debug CSV response exists and covers >= 90% of the chunk duration
    if debug_dir:
        chunk_csv_path = os.path.join(debug_dir, f"chunk_{chunk_idx:03d}_response.csv")
        if os.path.exists(chunk_csv_path):
            try:
                with open(chunk_csv_path, "r", encoding="utf-8") as f:
                    cached_text = f.read()
                cached_timed_elements = parse_csv_response(cached_text)
                
                if cached_timed_elements:
                    last_end = cached_timed_elements[-1].get("end", 0.0)
                    coverage = last_end / chunk_duration if chunk_duration > 0 else 0.0
                    
                    if coverage >= 0.90:
                        logging.info(f"Found valid cached response for chunk {chunk_idx:03d} (covers {coverage*100:.1f}% of chunk). Reusing cache.")
                        
                        # Shift all timestamps by the global chunk offset
                        shifted_elements = []
                        for el in cached_timed_elements:
                            shifted_elements.append({
                                "word": el.get("word", ""),
                                "start": round(el.get("start", 0.0) + offset, 2),
                                "end": round(el.get("end", 0.0) + offset, 2),
                                "type": el.get("type", "word")
                            })
                        chunk_transcript = " ".join([el["word"] for el in shifted_elements if el["type"] == "word"])
                        return shifted_elements, chunk_transcript
                    else:
                        logging.info(f"Existing cache for chunk {chunk_idx:03d} only covers {coverage*100:.1f}% (< 90%). Re-running transcription.")
            except Exception as cache_err:
                logging.warning(f"Failed to read cache for chunk {chunk_idx:03d} ({cache_err}). Re-running transcription...")

    logging.info(f"Uploading chunk to Gemini: {os.path.basename(chunk_path)} (Offset: {offset}s)")
    audio_file = client.files.upload(file=chunk_path)
    
    try:
        prompt = """
        You are an expert audio transcriber and phonetic analyst. The audio is spoken in Nepali (and may include English words or background noises).
        Provide a sequential timeline breakdown of this audio segment into discrete words, silence durations, or unpredicted sounds/fillers.
        For each element, specify:
        - word: the text representation of the word, silence descriptor (e.g., "[silence]"), or unpredicted sound/word.
        - start: start time in seconds relative to the beginning of this audio chunk (floating point).
        - end: end time in seconds relative to the beginning of this audio chunk (floating point).
        - type: classification of the item, such as "word", "silence", or "filler"
           
        Return your output strictly as a CSV string separated by pipes (|) with no surrounding markdown formatting if possible, for example:
        हामीले|4.888|5.444|word
        अर्को|5.500|6.111|word
        """
        
        attempt = 1
        while True:
            try:
                logging.info(f"Sending chunk request to Gemini (gemini-3.8-flash) [Attempt {attempt}]...")
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[audio_file, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=65536,
                    ),
                )
                break
            except APIError as e:
                if e.code == 503 or "UNAVAILABLE" in str(e):
                    logging.warning(f"Service unavailable. Waiting 60s before retry...")
                    time.sleep(65)
                    attempt += 1
                else:
                    raise e

        # Save individual chunk CSV response in debug mode
        if debug_dir:
            chunk_csv_path = os.path.join(debug_dir, f"chunk_{chunk_idx:03d}_response.csv")
            with open(chunk_csv_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            logging.info(f"Saved debug chunk response to: {chunk_csv_path}")

        chunk_timed_elements = parse_csv_response(response.text)
        
        # Shift all timestamps by the global chunk offset
        shifted_elements = []
        for el in chunk_timed_elements:
            shifted_elements.append({
                "word": el.get("word", ""),
                "start": round(el.get("start", 0.0) + offset, 2),
                "end": round(el.get("end", 0.0) + offset, 2),
                "type": el.get("type", "word")
            })
            
        chunk_transcript = " ".join([el["word"] for el in shifted_elements if el["type"] == "word"])
        return shifted_elements, chunk_transcript

    finally:
        try:
            client.files.delete(name=audio_file.name)
            logging.info(f"Cleaned up chunk from Gemini storage: {os.path.basename(chunk_path)}")
        except Exception:
            pass


def generate_fine_grained_transcription(audio_path: str, debug_mode: bool = True) -> Tuple[List[dict], str]:
    raw_transcript_path = audio_path.rsplit(".", 1)[0] + "_transcript.csv"
    
    debug_dir = None
    if debug_mode:
        base_dir = os.path.dirname(audio_path)
        debug_dir = os.path.join(base_dir, "debug_chunks")
        os.makedirs(debug_dir, exist_ok=True)
        
    if os.path.exists(raw_transcript_path):
        logging.info(f"Found existing fine-grained transcription cache: {raw_transcript_path}. Skipping Gemini inference.")
        try:
            timed_elements = []
            with open(raw_transcript_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter='|')
                for row in reader:
                    if len(row) >= 4 and not (row[0].strip().lower() == "word"):
                        timed_elements.append({
                            "word": row[0].strip(),
                            "start": float(row[1].strip()),
                            "end": float(row[2].strip()),
                            "type": row[3].strip()
                        })
            full_transcript = " ".join([el["word"] for el in timed_elements if el["type"] == "word"])
            return timed_elements, full_transcript
        except Exception as cache_err:
            logging.warning(f"Failed to read cache file ({cache_err}), proceeding with chunked processing...")

    logging.info("Initializing Google GenAI client for chunked transcription...")
    client = genai.Client()
    
    # Split audio into chunks with sliding overlap
    chunks = split_audio_into_chunks(audio_path, chunk_duration_sec=300, overlap_sec=30, debug_dir=debug_dir)
    
    all_timed_elements = []
    all_transcripts = []
    
    try:
        for idx, (chunk_path, offset) in enumerate(chunks):
            timed_elements, chunk_text = transcribe_single_chunk(client, chunk_path, offset, chunk_idx=idx, debug_dir=debug_dir)
            all_timed_elements.extend(timed_elements)
            if chunk_text:
                all_transcripts.append(chunk_text)
                
        full_transcript = " ".join(all_transcripts)
        
        logging.info(f"Saving combined fine-grained transcript data to: {raw_transcript_path}")
        with open(raw_transcript_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(["word", "start", "end", "type"])
            for el in all_timed_elements:
                writer.writerow([el["word"], el["start"], el["end"], el["type"]])
            
        return all_timed_elements, full_transcript
        
    except Exception as e:
        logging.error(f"Error during chunked transcription workflow: {e}")
        return [], ""


if __name__ == "__main__":
    audio_path = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.47.21 Lekh Poudel's Zoom Meeting\audio1103826994.m4a"
    timed_elements, transcript = generate_fine_grained_transcription(audio_path=audio_path, debug_mode=True)
    print(f"Successfully transcribed {len(timed_elements)} elements across all chunks.")