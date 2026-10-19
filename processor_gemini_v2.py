import os
import logging
from typing import List, Tuple, Optional, Callable
import json
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydub import AudioSegment, silence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_optimal_bitrate(video_path: str) -> str:
    """
    Programmatically reads video stream metadata using ffprobe 
    and determines an optimal target bitrate matching the input.
    """
    import subprocess
    
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,bit_rate",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        
        width = int(stream.get('width', 1920))
        height = int(stream.get('height', 1080))
        source_bitrate = int(stream.get('bit_rate', 0))
        
        # If ffprobe successfully extracted the source bitrate, match it closely
        if source_bitrate > 0:
            source_mbps = source_bitrate / 1_000_000
            target_bps = min(source_bitrate, 3_000_000) if height >= 1080 else min(source_bitrate, 1_500_000)
            target_str = f"{int(target_bps / 1000)}k"
            logging.info(f"Programmatic Bitrate: Detected {width}x{height} at {source_mbps:.2f} Mbps. Setting target to {target_str}.")
            return target_str
            
        # Fallback based on resolution if bit_rate metadata is missing
        if height >= 1080:
            logging.info(f"Programmatic Bitrate: Resolution is {height}p, defaulting to 2500k.")
            return "2500k"
        else:
            logging.info(f"Programmatic Bitrate: Resolution is {height}p, defaulting to 1200k.")
            return "1200k"
            
    except Exception as e:
        logging.warning(f"Could not programmatically detect bitrate ({e}), falling back to 1500k.")
        return "1500k"
    
def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.6f}"


def extract_audio(video_path: str, audio_path: str):
    """Extracts audio from video file. Returns True if audio exists, False otherwise."""
    import subprocess

    logging.info(f"Extracting audio from {video_path} to {audio_path}")
    
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name", "-of", "json", video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if not data.get("streams"):
            logging.warning("Video has no audio track - skipping audio-based detection")
            return False

        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e}")
        return False

def detect_silence(audio_path: str, min_silence_len: int = 2000, silence_thresh: int = -40) -> List[Tuple[float, float]]:
    """Detects silence in audio file and returns list of (start, end) tuples in seconds."""
    logging.info("Detecting silence...")
    audio = AudioSegment.from_file(audio_path)
    silence_intervals_ms = silence.detect_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )
    silence_intervals_sec = [(start / 1000, end / 1000) for start, end in silence_intervals_ms]
    
    if silence_intervals_sec:
        logging.info(f"Found {len(silence_intervals_sec)} silence intervals:")
        for i, (start, end) in enumerate(silence_intervals_sec, 1):
            duration = end - start
            logging.info(f"  Silence {i}: {format_timestamp(start)} - {format_timestamp(end)} (duration: {duration:.6f}s)")
    else:
        logging.info("No silence intervals found.")
    
    return silence_intervals_sec

def detect_filler_words(audio_path: str, filler_words_list: List[str] = None) -> Tuple[List[Tuple[float, float]], str]:
    """Detects filler words and transcribes audio using Google Gemini API (gemini-3.8-flash) or loads from cache if available."""
    raw_transcript_path = audio_path.split(".")[0] + "_raw_transcript.json"
    transcript_path = audio_path.replace(".wav", "_transcript.txt")
    
    # Check if raw transcript JSON already exists to skip Gemini inference
    if os.path.exists(raw_transcript_path):
        logging.info(f"Found existing raw transcript cache: {raw_transcript_path}. Skipping Gemini inference.")
        try:
            with open(raw_transcript_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                
            transcript_text = result_data.get("transcript", "")
            # Fallback if transcript text file also exists
            if not transcript_text and os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as tf:
                    transcript_text = tf.read()
                    
            raw_fillers = result_data.get("filler_words", [])
            filler_intervals = []
            seen_intervals = set()
            
            for item in raw_fillers:
                word_text = item.get("word", "").strip()
                start_time = item.get("start")
                end_time = item.get("end")
                
                if start_time is None or end_time is None or end_time <= start_time:
                    continue
                    
                interval_key = (round(float(start_time), 2), round(float(end_time), 2))
                if interval_key in seen_intervals:
                    continue
                seen_intervals.add(interval_key)
                
                duration = float(end_time) - float(start_time)
                filler_intervals.append((float(start_time), float(end_time)))
                logging.info(f"    ✓ [Cached] Filler word detected: '{word_text}' at {format_timestamp(float(start_time))} - {format_timestamp(float(end_time))} (duration: {duration:.2f}s)")
            
            logging.info(f"Loaded {len(filler_intervals)} unique filler words total from cache.")
            return filler_intervals, transcript_text
            
        except Exception as cache_err:
            logging.warning(f"Failed to read cache file ({cache_err}), proceeding with Gemini API call...")

    # Otherwise, proceed with standard Gemini API call
    logging.info("Initializing Google GenAI client for transcription and filler word detection...")
    client = genai.Client()
    
    if filler_words_list:
        filler_words = [w.lower().strip() for w in filler_words_list]
        logging.info(f"Using custom filler words: {filler_words}")
    else:
        filler_words = ["um", "uh", "umm", "uhh", "er", "just", "you know", "like"]
        logging.info(f"Using default filler words: {filler_words}")
        
    logging.info(f"Uploading audio file to Gemini: {audio_path}")
    audio_file = client.files.upload(file=audio_path)
    
    try:
        prompt = f"""
        You are an expert audio transcriber and editor. 
        1. Provide the full verbatim transcript of the audio file.
        2. Identify all occurrences of filler words from this list: {filler_words}.
        For each detected filler word, provide its precise start time (in seconds), end time (in seconds), and the word itself.
        
        Return your output strictly as a JSON object with two fields:
        - "transcript": string containing the full transcript text.
        - "filler_words": a list of objects, each with keys "word", "start", and "end" (floating point numbers representing seconds).
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
                    ),
                )
                break  # Successful response, exit loop
                
            except APIError as e:
                # Check for 503 Service Unavailable / Overloaded error codes
                if e.code == 503 or "503" in str(e) or "UNAVAILABLE" in str(e):
                    logging.warning(f"Gemini API returned 503 Service Unavailable (Attempt {attempt}). Waiting for 1 minute before retrying...")
                    time.sleep(60)
                    attempt += 1
                else:
                    raise e  # Raise other API errors immediately
            except Exception as e:
                # Handle connection errors or temporary network glitches that may return 503-like symptoms
                err_str = str(e)
                if "503" in err_str or "Service Unavailable" in err_str or "UNAVAILABLE" in err_str:
                    logging.warning(f"Encountered service interruption (Attempt {attempt}): {e}. Waiting for 1 minute before retrying...")
                    time.sleep(60)
                    attempt += 1
                else:
                    raise e

        result_data = json.loads(response.text)
        
        logging.info(f"Saving raw transcript and filler word data to: {raw_transcript_path}")
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4)
            
        transcript_text = result_data.get("transcript", "")
        raw_fillers = result_data.get("filler_words", [])
        
        filler_intervals = []
        seen_intervals = set()
        
        for item in raw_fillers:
            word_text = item.get("word", "").strip()
            start_time = item.get("start")
            end_time = item.get("end")
            
            if start_time is None or end_time is None or end_time <= start_time:
                continue
                
            interval_key = (round(float(start_time), 2), round(float(end_time), 2))
            if interval_key in seen_intervals:
                continue
            seen_intervals.add(interval_key)
            
            duration = float(end_time) - float(start_time)
            filler_intervals.append((float(start_time), float(end_time)))
            logging.info(f"    ✓ Filler word detected: '{word_text}' at {format_timestamp(float(start_time))} - {format_timestamp(float(end_time))} (duration: {duration:.2f}s)")
        
        logging.info(f"Found {len(filler_intervals)} unique filler words total via Gemini.")
        
        if transcript_text:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            logging.info(f"Transcript saved to: {transcript_path}")
            
        return filler_intervals, transcript_text
        
    except Exception as e:
        logging.error(f"Error during Gemini transcription/filler detection: {e}")
        return [], ""
        
    finally:
        try:
            client.files.delete(name=audio_file.name)
            logging.info("Cleaned up audio file from Gemini storage.")
        except Exception as cleanup_err:
            logging.warning(f"Failed to delete remote audio file: {cleanup_err}")

def detect_freeze_frames(video_path: str, min_duration: float = 5.0, noise_tolerance: float = 0.001) -> List[Tuple[float, float]]:
    """Detects freeze/still frames using FFmpeg's freezedetect filter."""
    import subprocess
    import re
    
    logging.info(f"Detecting freeze frames (min duration: {min_duration}s, noise tolerance: {noise_tolerance})...")
    
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'freezedetect=n={noise_tolerance}:d={min_duration}',
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        freeze_intervals = []
        freeze_start = None
        
        for line in result.stderr.split('\n'):
            if 'freezedetect' in line:
                if 'freeze_start' in line:
                    match = re.search(r'freeze_start:\s*([\d.]+)', line)
                    if match:
                        freeze_start = float(match.group(1))
                elif 'freeze_end' in line and freeze_start is not None:
                    match = re.search(r'freeze_end:\s*([\d.]+)', line)
                    if match:
                        freeze_end = float(match.group(1))
                        freeze_intervals.append((freeze_start, freeze_end))
                        freeze_start = None
        
        if freeze_intervals:
            logging.info(f"Found {len(freeze_intervals)} freeze frame intervals:")
            for i, (start, end) in enumerate(freeze_intervals, 1):
                duration = end - start
                logging.info(f"  Freeze {i}: {format_timestamp(start)} - {format_timestamp(end)} (duration: {duration:.2f}s)")
        else:
            logging.info("No freeze frames detected.")
        
        return freeze_intervals
        
    except subprocess.TimeoutExpired:
        logging.error("Freeze detection timed out")
        return []
    except Exception as e:
        logging.error(f"Error detecting freeze frames: {e}")
        return []

def merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.1) -> List[Tuple[float, float]]:
    """Merges overlapping or close intervals."""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        
        if current_start <= last_end + min_gap:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
            
    return merged

def invert_intervals(intervals: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    """Inverts intervals to get parts to KEEP."""
    keep_intervals = []
    current_time = 0.0
    
    for start, end in intervals:
        if start > current_time:
            keep_intervals.append((current_time, start))
        current_time = max(current_time, end)
        
    if current_time < total_duration:
        keep_intervals.append((current_time, total_duration))
    return keep_intervals

def extract_segments_ffmpeg(input_path: str, segments: List[Tuple[float, float]], temp_dir: str = ".", target_bitrate: str = "4M", progress_callback: Optional[Callable[[int, str], None]] = None) -> List[str]:
    """Extract video segments using FFmpeg with re-encoding for precise cuts."""
    import subprocess
    
    segment_files = []
    
    for i, (start, end) in enumerate(segments):
        duration = end - start
        segment_file = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
        
        logging.info(f"Extracting segment {i+1}/{len(segments)}: {format_timestamp(start)} - {format_timestamp(end)}")
        if progress_callback:
            progress = 60 + int(30 * ((i + 1) / len(segments)))
            progress_callback(progress, f"Extracting segment {i+1}/{len(segments)}")
        
        try:
            cmd = [
                'ffmpeg',
                '-accurate_seek',
                '-i', input_path,
                '-ss', str(start),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-b:v', target_bitrate,
                '-maxrate', target_bitrate,
                '-bufsize', f'{int(target_bitrate.rstrip("kM")) * 2}M' if 'M' in target_bitrate else f'{int(target_bitrate.rstrip("k")) * 2}k',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-avoid_negative_ts', '1',
                '-y',
                segment_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logging.error(f"FFmpeg segment extraction failed: {result.stderr}")
                raise RuntimeError(f"Failed to extract segment {i}")
            
            segment_files.append(segment_file)
            
        except Exception as e:
            logging.error(f"Error extracting segment {i}: {e}")
            for f in segment_files:
                if os.path.exists(f):
                    os.remove(f)
            raise
            
    return segment_files

def concatenate_segments_ffmpeg(segment_files: List[str], output_path: str, codec: str, preset: str, ffmpeg_params: list, bitrate: str = None) -> None:
    """Concatenate video segments using FFmpeg."""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='.') as f:
        concat_file = f.name
        for segment in segment_files:
            escaped_path = segment.replace('\\', '/').replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    try:
        logging.info(f"Concatenating {len(segment_files)} segments with FFmpeg...")
        cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file]
        
        if codec == 'copy':
            cmd.extend(['-c', 'copy'])
        else:
            cmd.extend(['-c:v', codec, '-c:a', 'aac'])
            if bitrate:
                cmd.extend(['-b:v', bitrate])
            if preset:
                cmd.extend(['-preset', preset])
            if ffmpeg_params:
                cmd.extend(ffmpeg_params)
        
        cmd.extend(['-y', output_path])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg concatenation failed: {result.stderr}")
            raise RuntimeError("Failed to concatenate segments")
            
    finally:
        if os.path.exists(concat_file):
            os.remove(concat_file)

def get_encoding_params(use_gpu: bool, use_crf: bool, bitrate: str, crf: int, preset: str):
    """Get encoding parameters for CPU or GPU encoding."""
    if use_gpu:
        import subprocess
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=5)
            if 'h264_nvenc' not in result.stdout:
                use_gpu = False
        except Exception:
            use_gpu = False
            
    if use_gpu:
        codec = "h264_nvenc"
        nvenc_preset_map = {"veryslow": "hq", "slow": "hq", "medium": "medium", "fast": "fast", "ultrafast": "hp"}
        return codec, nvenc_preset_map.get(preset, "medium"), []
    else:
        codec = "libx264"
        if use_crf:
            ffmpeg_params = ['-crf', str(crf), '-maxrate', bitrate, '-bufsize', f'{int(bitrate.rstrip("k")) * 2}k']
        else:
            ffmpeg_params = ['-b:v', bitrate]
        return codec, preset, ffmpeg_params

def get_video_rotation(video_path: str) -> int:
    """Get rotation metadata using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream_tags=rotate', '-of', 'default=nw=1:nk=1', video_path],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip()) if result.stdout.strip() else 0
    except Exception:
        return 0

def transpose_video_if_needed(input_path: str, rotation: int) -> str:
    """Transpose video if rotation metadata is present."""
    if rotation == 0:
        return input_path
    import subprocess
    import tempfile
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='transposed_')
    os.close(temp_fd)
    
    transpose_map = {90: '2', 180: '2,transpose=2', 270: '1'}
    transpose_filter = transpose_map.get(rotation)
    if not transpose_filter:
        return input_path
        
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-vf', f'transpose={transpose_filter}', '-c:a', 'copy', '-y', temp_path], capture_output=True, check=True, timeout=1800)
        return temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return input_path

def process_video(input_path: str, output_path: str, min_silence_len: int = 2000, silence_thresh: int = -63, crossfade_duration: float = 0.2, bitrate: Optional[str] = None, crf: int = 18, preset: str = "medium", use_crf: bool = False, use_gpu_encoding: bool = False, no_crossfade: bool = False, filler_words: List[str] = None, freeze_duration: float = None, freeze_noise: float = 0.001, remove_background: bool = False, bg_color: str = "green", bg_image: Optional[str] = None, rvm_model: str = "mobilenetv3", rvm_downsample: Optional[float] = None, use_segmentation: bool = False, seg_model: str = "general", seg_threshold: float = 0.5, seg_smooth: int = 5, rvm_erode: int = 0, rvm_dilate: int = 0, rvm_median: int = 0, rvm_blur: int = 0, progress_callback: Optional[Callable[[int, str], None]] = None, remove_silence: bool = True):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return {'status': 'error', 'error': 'Input file not found'}
    
    rotation = get_video_rotation(input_path)
    working_video_path = transpose_video_if_needed(input_path, rotation)
    transposed = (working_video_path != input_path)
    temp_audio_path = working_video_path.split(".")[0] + "_temp_audio.wav"
    transcript_text = None
    
    if progress_callback:
        progress_callback(0, "Starting processing...")
    
    try:
        # Automatically determine bitrate if not passed explicitly
        if not bitrate:
            bitrate = get_optimal_bitrate(working_video_path)
        logging.info(f"Target working bitrate set to: {bitrate}")

        if progress_callback:
            progress_callback(5, "Extracting audio...")
        has_audio = extract_audio(working_video_path, temp_audio_path)
        
        silence_intervals = []
        if has_audio and remove_silence:
            if progress_callback:
                progress_callback(10, "Detecting silence...")
            silence_intervals = detect_silence(temp_audio_path, min_silence_len, silence_thresh)
            
        filler_intervals = []
        if has_audio:
            if progress_callback:
                progress_callback(20, "Detecting filler words and transcribing via Gemini (or cache)...")
            filler_intervals, transcript_text = detect_filler_words(temp_audio_path, filler_words_list=filler_words)
            
        freeze_intervals = []
        if freeze_duration is not None and freeze_duration > 0:
            if progress_callback:
                progress_callback(40, "Detecting freeze frames...")
            freeze_intervals = detect_freeze_frames(working_video_path, min_duration=freeze_duration, noise_tolerance=freeze_noise)
            
        all_remove_intervals = silence_intervals + filler_intervals + freeze_intervals
        merged_remove_intervals = merge_intervals(all_remove_intervals)
        
        import subprocess
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration:stream=width,height", "-of", "json", working_video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        total_duration = float(probe_data['format']['duration'])
        keep_intervals = invert_intervals(merged_remove_intervals, total_duration)
        
        if len(keep_intervals) == 1 and keep_intervals[0] == (0.0, total_duration):
            logging.info("✅ No cuts needed - no target intervals detected.")
            if remove_background or use_segmentation:
                import shutil
                shutil.copy2(working_video_path, output_path)
            else:
                return {'status': 'skipped', 'transcript': transcript_text}

        segment_files = []
        try:
            if progress_callback:
                progress_callback(60, "Extracting video segments...")
            segment_files = extract_segments_ffmpeg(working_video_path, keep_intervals, temp_dir=".", target_bitrate=bitrate, progress_callback=progress_callback)
            
            codec, preset_value, ffmpeg_params = get_encoding_params(use_gpu_encoding, use_crf, bitrate, crf, preset)
            
            if no_crossfade:
                concatenate_segments_ffmpeg(segment_files, output_path, codec='copy', preset=None, ffmpeg_params=[], bitrate=None)
            else:
                concatenate_segments_ffmpeg(segment_files, output_path, codec=codec, preset=preset_value, ffmpeg_params=ffmpeg_params, bitrate=bitrate if use_gpu_encoding else None)
            
            logging.info(f"✅ Video processing complete: {output_path}")
            
        finally:
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    os.remove(segment_file)
                    
        return {'status': 'complete', 'output_path': output_path, 'transcript': transcript_text}

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'status': 'error', 'error': str(e)}
        
    finally:
        if transposed and os.path.exists(working_video_path):
            os.remove(working_video_path)


if __name__ == "__main__":
    input_test_file = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 22.50.40 Lekh Poudel's Zoom Meeting\video1986872404.mp4"
    input_test_file = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.47.21 Lekh Poudel's Zoom Meeting\video1103826994.mp4"
    output_test_file = input_test_file.replace(".mp4", "_edited.mp4")
    
    if os.path.exists(input_test_file):
        print(f"▶️ Running standalone processing on: {input_test_file}")
        
        process_video(
            input_path=input_test_file,
            output_path=output_test_file,
            min_silence_len=900,
            silence_thresh=-64,
            remove_silence=True,
            crossfade_duration=0.2,
            preset="medium",
            use_crf=False,
            crf=18,
            filler_words=["um", "uh", "umm", "uhh", "er", "like", "just, you know", "like", "अं",
            "आं", "अँ", "आँ", "लाइक", "हँ", "ए", "लाइग", "अनि", "मतलब", "हैत", "अ"],
            freeze_duration=3.0,
            remove_background=False
        )
    else:
        print(f"ℹ️ Test video path not found.")