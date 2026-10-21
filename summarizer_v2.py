import os
import json
import time
import logging
from typing import List
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PodcastHighlightExtractor:
    def __init__(self, transcript_path: str, target_duration_sec: int = 120, model_name: str = "gemini-3.8-flash"):
        self.transcript_path = transcript_path
        self.target_duration = target_duration_sec
        self.model_name = model_name
        self.client = genai.Client()

    def analyze_and_extract_highlights(self) -> dict:
        """
        Uploads the transcript file directly via the Gemini Files API, 
        analyzes it using gemini-3.8-flash, and returns the highlight segments JSON.
        """
        if not os.path.exists(self.transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_path}")

        logging.info(f"Uploading transcript file to Gemini: {self.transcript_path}")
        uploaded_file = self.client.files.upload(file=self.transcript_path)
        
        try:
            prompt = f"""
            You are an expert audio editor and content producer. 
            Analyze the attached podcast transcript file (which includes words and timestamps). 
            Your task is to select key thematic highlight segments that capture the most interesting and impactful parts of the conversation.

            Constraints:
            1. Total cumulative duration of all selected segments should be close to approximately {self.target_duration} seconds.
            2. Each segment should be approximately 5-20 seconds long. Each segment must start and end on meaningful statement on their own. 
            3. The segment should reflect the main ideas of the conversation, avoiding trivial or filler content. Or something that can potential go viral on social media.
            4. Provide precise start and end timestamps (in seconds) matching the transcript data.
            5. Output ONLY a valid JSON object matching this schema:
            {{
                "total_highlight_duration_seconds": <float>,
                "segments": [
                    {{
                        "topic": "<string describing the theme>",
                        "start_time": <float>,
                        "end_time": <float>,
                        "transcript_text": "<the exact text of the segment>"
                    }}
                ]
            }}
            """

            max_retries = 10
            wait_time = 45  # Start with a safe initial wait for large 1MB token windows

            for attempt in range(1, max_retries + 1):
                try:
                    logging.info(f"Sending request to Gemini using model '{self.model_name}' [Attempt {attempt}/{max_retries}]...")
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[uploaded_file, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2
                        ),
                    )
                    return json.loads(response.text)
                
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                    is_unavailable = "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str

                    if is_rate_limit or is_unavailable:
                        # Try to extract exact server retry delay if provided
                        try:
                            if hasattr(e, 'details') and e.details:
                                for detail in e.details:
                                    if isinstance(detail, dict) and 'retryDelay' in detail:
                                        delay_val = detail['retryDelay'].lower().replace('s', '').strip()
                                        wait_time = max(wait_time, float(delay_val))
                        except Exception:
                            pass

                        if attempt < max_retries:
                            logging.warning(f"⚠️ Free tier token/server limit hit. Waiting {wait_time:.1f}s for quota window to reset... (Attempt {attempt}/{max_retries})")
                            time.sleep(wait_time)
                            wait_time = min(wait_time * 1.5, 90)  # Scale up wait time safely
                            continue
                    
                    logging.error(f"❌ Error during highlight extraction: {error_str}")
                    raise e
        finally:
            try:
                logging.info("Cleaning up uploaded transcript file from Gemini storage...")
                self.client.files.delete(name=uploaded_file.name)
            except Exception as cleanup_err:
                logging.warning(f"Failed to delete remote file: {cleanup_err}")

    

def get_optimal_bitrate(video_path: str) -> str:
    import subprocess
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,bit_rate", "-of", "json", video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        height = int(stream.get('height', 1080))
        source_bitrate = int(stream.get('bit_rate', 0))
        
        if source_bitrate > 0:
            target_bps = min(source_bitrate, 3_000_000) if height >= 1080 else min(source_bitrate, 1_500_000)
            return f"{int(target_bps / 1000)}k"
        return "2500k" if height >= 1080 else "1200k"
    except Exception:
        return "1500k"


def extract_highlight_segments_ffmpeg(input_path: str, segments: List[dict], output_path: str, bitrate: str = "2500k"):
    import subprocess
    import tempfile

    segment_files = []
    temp_dir = "."
    
    try:
        for i, seg in enumerate(segments):
            start = seg["start_time"]
            end = seg["end_time"]
            duration = end - start
            segment_file = os.path.join(temp_dir, f"highlight_seg_{i:04d}.mp4")
            
            cmd = [
                'ffmpeg', '-accurate_seek', '-i', input_path,
                '-ss', str(start), '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'fast', '-b:v', bitrate,
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', '1', '-y', segment_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to extract highlight segment {i}: {result.stderr}")
            segment_files.append(segment_file)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=temp_dir) as f:
            concat_file = f.name
            for segment in segment_files:
                escaped_path = segment.replace('\\', '/').replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        try:
            concat_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c:v', 'libx264', '-preset', 'fast', '-b:v', bitrate,
                '-c:a', 'aac', '-y', output_path
            ]
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to concatenate highlight clips: {result.stderr}")
            logging.info(f"✅ Summary highlight clip successfully created: {output_path}")
        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)

    finally:
        for sf in segment_files:
            if os.path.exists(sf):
                os.remove(sf)


def generate_podcast_summary_clip(video_path: str, transcript_json_path: str, output_summary_path: str, target_duration_sec: int = 120):
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return
    if not os.path.exists(transcript_json_path):
        logging.error(f"Transcript JSON file not found: {transcript_json_path}")
        return

    logging.info(f"▶️ Starting summary generation pipeline...")
    logging.info(f"   Video: {video_path}")
    logging.info(f"   Transcript: {transcript_json_path}")

    try:
        extractor = PodcastHighlightExtractor(
            transcript_path=transcript_json_path,
            target_duration_sec=target_duration_sec,
            model_name="gemini-3.8-flash"
        )
        
        logging.info("Analyzing transcript via Files API to extract key highlight intervals...")
        highlights = extractor.analyze_and_extract_highlights()
        
        highlight_json_output = video_path.rsplit(".", 1)[0] + "_highlight_segments.json"
        with open(highlight_json_output, "w", encoding="utf-8") as f:
            json.dump(highlights, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved highlight segments JSON to {highlight_json_output}")

        segments = highlights.get("segments", [])
        if not segments:
            logging.warning("No segments returned by Gemini summarizer.")
            return

        bitrate = get_optimal_bitrate(video_path)
        extract_highlight_segments_ffmpeg(video_path, segments, output_summary_path, bitrate=bitrate)

    except Exception as e:
        logging.error(f"An error occurred during summary generation: {e}")


if __name__ == "__main__":
    input_video = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.47.21 Lekh Poudel's Zoom Meeting\video1103826994.mp4"
    transcript_file = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.47.21 Lekh Poudel's Zoom Meeting\audio1103826994_transcript.json"
    output_summary = input_video.replace(".mp4", "_summary_clip.mp4")
    
    if os.path.exists(input_video) and os.path.exists(transcript_file):
        generate_podcast_summary_clip(
            video_path=input_video,
            transcript_json_path=transcript_file,
            output_summary_path=output_summary,
            target_duration_sec=120
        )
    else:
        print("ℹ️ Input video or transcript file path not found. Please verify your file paths.")