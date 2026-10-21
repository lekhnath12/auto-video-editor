import os
import logging
from typing import List, Tuple, Optional, Callable
import json
import time

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
        
        if source_bitrate > 0:
            source_mbps = source_bitrate / 1_000_000
            target_bps = min(source_bitrate, 3_000_000) if height >= 1080 else min(source_bitrate, 1_500_000)
            target_str = f"{int(target_bps / 1000)}k"
            logging.info(f"Programmatic Bitrate: Detected {width}x{height} at {source_mbps:.2f} Mbps. Setting target to {target_str}.")
            return target_str
            
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

def extract_audio(video_path: str, audio_path: str) -> Optional[str]:
    """Extracts compressed audio from video file for minimal file size. Returns the actual audio path or None."""
    import subprocess

    logging.info(f"Extracting compressed audio from {video_path} to {audio_path}")
    
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name", "-of", "json", video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if not data.get("streams"):
            logging.warning("Video has no audio track - skipping audio-based detection")
            return None

        if audio_path.endswith('.wav'):
            audio_path = audio_path.replace('.wav', '.m4a')

        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn", 
            "-acodec", "aac", 
            "-b:a", "64k",     
            "-ar", "16000",    
            "-ac", "1",        
            audio_path
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e}")
        return None
    
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

def load_deleted_words_intervals(json_path: str) -> List[Tuple[float, float]]:
    """Loads intervals to remove from a JSON file containing words with start_time and end_time."""
    logging.info(f"Loading deleted words intervals from: {json_path}")
    if not os.path.exists(json_path):
        logging.warning(f"Deleted words JSON file not found at {json_path}. Skipping word deletion.")
        return []
        
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        intervals = []
        seen_intervals = set()
        
        for item in data:
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            
            if start_time is None or end_time is None or end_time <= start_time:
                continue
                
            interval_key = (round(float(start_time), 2), round(float(end_time), 2))
            if interval_key in seen_intervals:
                continue
            seen_intervals.add(interval_key)
            
            intervals.append((float(start_time), float(end_time)))
            
        logging.info(f"Loaded {len(intervals)} unique intervals from {json_path}.")
        return intervals
    except Exception as e:
        logging.error(f"Failed to load or parse deleted words JSON file: {e}")
        return []

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
        
        return freeze_intervals
        
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

def extract_segments_ffmpeg(input_path: str, segments: List[Tuple[float, float]], temp_dir: str = ".", target_bitrate: str = "4M", use_gpu_encoding: bool = False, progress_callback: Optional[Callable[[int, str], None]] = None) -> List[str]:
    """Extract video segments using FFmpeg with optimized speed presets (fast / nvenc)."""
    import subprocess
    
    segment_files = []
    codec, preset_val, extra_params = get_encoding_params(use_gpu_encoding, False, target_bitrate, 18, "fast")
    
    for i, (start, end) in enumerate(segments):
        duration = end - start
        segment_file = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
        
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
                '-c:v', codec,
                '-preset', preset_val,
            ]
            cmd.extend(extra_params)
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-avoid_negative_ts', '1',
                '-y',
                segment_file
            ])
            
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
    """Get optimized encoding parameters for speed."""
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
        return codec, "p4", ['-b:v', bitrate]
    else:
        codec = "libx264"
        return codec, "fast", ['-b:v', bitrate]

def get_video_rotation(video_path: str) -> int:
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

def process_video(input_path: str, output_path: str, min_silence_len: int = 2000, silence_thresh: int = -63, crossfade_duration: float = 0.2, bitrate: Optional[str] = None, crf: int = 18, preset: str = "fast", use_crf: bool = False, use_gpu_encoding: bool = False, no_crossfade: bool = False, deleted_words_json: str = "deleted_words.json", freeze_duration: float = None, freeze_noise: float = 0.001, remove_background: bool = False, bg_color: str = "green", bg_image: Optional[str] = None, progress_callback: Optional[Callable[[int, str], None]] = None, remove_silence: bool = True):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return {'status': 'error', 'error': 'Input file not found'}
    
    rotation = get_video_rotation(input_path)
    working_video_path = transpose_video_if_needed(input_path, rotation)
    transposed = (working_video_path != input_path)
    temp_audio_path = input_path.rsplit(".", 1)[0] + "_temp_audio.wav"
    
    if progress_callback:
        progress_callback(0, "Starting processing...")
    
    try:
        if not bitrate:
            bitrate = get_optimal_bitrate(working_video_path)
        logging.info(f"Target working bitrate set to: {bitrate}")

        has_audio = False
        silence_intervals = []
        if remove_silence:
            if progress_callback:
                progress_callback(5, "Extracting audio for silence detection...")
            temp_audio_path = extract_audio(working_video_path, temp_audio_path)
            has_audio = temp_audio_path is not None
            
            if has_audio:
                if progress_callback:
                    progress_callback(10, "Detecting silence...")
                silence_intervals = detect_silence(temp_audio_path, min_silence_len, silence_thresh)
            
        if progress_callback:
            progress_callback(20, "Loading deleted words intervals from JSON...")
        deleted_intervals = load_deleted_words_intervals(deleted_words_json)
            
        freeze_intervals = []
        if freeze_duration is not None and freeze_duration > 0:
            if progress_callback:
                progress_callback(40, "Detecting freeze frames...")
            freeze_intervals = detect_freeze_frames(working_video_path, min_duration=freeze_duration, noise_tolerance=freeze_noise)
            
        all_remove_intervals = silence_intervals + deleted_intervals + freeze_intervals
        merged_remove_intervals = merge_intervals(all_remove_intervals)
        
        import subprocess
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration:stream=width,height", "-of", "json", working_video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        total_duration = float(probe_data['format']['duration'])
        keep_intervals = invert_intervals(merged_remove_intervals, total_duration)
        
        if len(keep_intervals) == 1 and keep_intervals[0] == (0.0, total_duration):
            logging.info("✅ No cuts needed - no target intervals detected.")
            return {'status': 'skipped'}

        segment_files = []
        try:
            if progress_callback:
                progress_callback(60, "Extracting video segments...")
            segment_files = extract_segments_ffmpeg(working_video_path, keep_intervals, temp_dir=".", target_bitrate=bitrate, use_gpu_encoding=use_gpu_encoding, progress_callback=progress_callback)
            
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
                    
        return {'status': 'complete', 'output_path': output_path}

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'status': 'error', 'error': str(e)}
        
    finally:
        if transposed and os.path.exists(working_video_path):
            os.remove(working_video_path)
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


if __name__ == "__main__":
    input_test_file = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.20.58 Interview With Dr. Khadka\video1184682387.mp4"
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
            preset="fast",
            use_gpu_encoding=False,
            use_crf=False,
            crf=18,
            deleted_words_json="deleted_words.json",
            freeze_duration=3.0,
            remove_background=False
        )
    else:
        print(f"ℹ️ Test video path not found.")