import os
import logging
import json
from typing import List, Tuple, Optional, Callable

from transformers import pipeline
from pydub import AudioSegment, silence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


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

        # Extract audio using ffmpeg (Whisper requires 16000 Hz and mono -ac 1)
        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", audio_path
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e}")
        return False

def detect_silence(audio_path: str, min_silence_len: int = 2000, silence_thresh: int = -40) -> List[Tuple[float, float]]:
    """
    Detects silence in audio file.
    """
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
            logging.info(f"  Silence {i}: {format_timestamp(start)} - {format_timestamp(end)} (duration: {duration:.2f}s)")
    else:
        logging.info("No silence intervals found.")
    
    return silence_intervals_sec

def detect_filler_words(audio_path: str, model_size: str = "aamoshdahal/whisper-small-nepali", filler_words_list: List[str] = None) -> Tuple[List[Tuple[float, float]], str]:
    """
    Detects and logs filler words/transcript using Hugging Face pipeline.
    Returns an empty list for filler cuts to prevent accidental sentence deletion.
    """
    return detect_filler_words_whisper(audio_path, model_size, filler_words_list)

def detect_filler_words_whisper(audio_path: str, model_size: str = "aamoshdahal/whisper-small-nepali", filler_words_list: List[str] = None) -> Tuple[List[Tuple[float, float]], str]:
    """
    Transcribes audio using fine-tuned Hugging Face Nepali Whisper model.
    """
    logging.info(f"Loading Hugging Face model ({model_size})...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_size,
        device=-1
    )
    
    logging.info("Transcribing audio for Nepali transcription...")
    
    result = pipe(
        audio_path,
        chunk_length_s=30,
        return_timestamps=True,
        generate_kwargs={"language": "ne", "task": "transcribe"}
    )

    logging.info("Transcription complete. Processing results...")
    transcript_text = result.get("text", "")
    logging.info(f"Transcribed text: {transcript_text}")
    
    # Save transcript text to file
    if transcript_text:
        transcript_path = audio_path.replace(".wav", "_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        logging.info(f"Transcript text saved to: {transcript_path}")

    # Save raw ASR result (chunks and timestamps) to JSON file
    json_path = audio_path.replace(".wav", "_transcript.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logging.info(f"Transcript JSON saved to: {json_path}")
    except Exception as e:
        logging.error(f"Failed to save transcript JSON: {e}")
    
    # Return empty list for cuts to protect video integrity against chunk-level over-removal
    return [], transcript_text

def detect_freeze_frames(video_path: str, min_duration: float = 5.0, noise_tolerance: float = 0.001) -> List[Tuple[float, float]]:
    """
    Detects freeze/still frames using FFmpeg's freezedetect filter.
    """
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
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
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
    """Inverts intervals to get the parts we want to KEEP."""
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
                '-bufsize', f'{int(target_bitrate[:-1])*2}M',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-avoid_negative_ts', '1',
                '-y',
                segment_file
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
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
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
    finally:
        if os.path.exists(concat_file):
            os.remove(concat_file)

def get_encoding_params(use_gpu: bool, use_crf: bool, bitrate: str, crf: int, preset: str):
    """Get encoding parameters for CPU or GPU encoding."""
    if use_gpu:
        codec = "h264_nvenc"
        return codec, "medium", []
    else:
        codec = "libx264"
        if use_crf:
            ffmpeg_params = ['-crf', str(crf), '-maxrate', bitrate, '-bufsize', f'{int(bitrate.rstrip("k")) * 2}k']
        else:
            ffmpeg_params = ['-b:v', bitrate]
        return codec, preset, ffmpeg_params

def get_video_rotation(video_path: str) -> int:
    """Get rotation metadata from video."""
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_tags=rotate', '-of', 'default=nw=1:nk=1',
             video_path],
            capture_output=True, text=True, timeout=5
        )
        rotation = result.stdout.strip()
        return int(rotation) if rotation else 0
    except Exception:
        return 0

def transpose_video_if_needed(input_path: str, rotation: int) -> str:
    """Transpose video if rotation metadata exists."""
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
        subprocess.run(
            ['ffmpeg', '-i', input_path, '-vf', f'transpose={transpose_filter}', '-c:a', 'copy', '-y', temp_path],
            capture_output=True, check=True, timeout=1800
        )
        return temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return input_path

def process_video(input_path: str, output_path: str, min_silence_len: int = 2000, silence_thresh: int = -63, crossfade_duration: float = 0.2, bitrate: str = "5000k", crf: int = 18, preset: str = "medium", use_crf: bool = False, use_gpu_encoding: bool = False, no_crossfade: bool = False, filler_words: List[str] = None, freeze_duration: float = None, freeze_noise: float = 0.001, remove_background: bool = False, bg_color: str = "green", bg_image: Optional[str] = None, rvm_model: str = "mobilenetv3", rvm_downsample: Optional[float] = None, use_segmentation: bool = False, seg_model: str = "general", seg_threshold: float = 0.5, seg_smooth: int = 5, rvm_erode: int = 0, rvm_dilate: int = 0, rvm_median: int = 0, rvm_blur: int = 0, progress_callback: Optional[Callable[[int, str], None]] = None, remove_silence: bool = True):
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
        
    rotation = get_video_rotation(input_path)
    working_video_path = transpose_video_if_needed(input_path, rotation)
    transposed = (working_video_path != input_path)

    temp_audio_path = "temp_audio.wav"
    transcript_text = None
    
    if progress_callback:
        progress_callback(0, "Starting processing...")
        
    try:
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
                progress_callback(20, "Transcribing with Hugging Face Nepali model...")
            filler_intervals, transcript_text = detect_filler_words(temp_audio_path, filler_words_list=filler_words)
            
        freeze_intervals = []
        if freeze_duration is not None and freeze_duration > 0:
            if progress_callback:
                progress_callback(40, "Detecting freeze frames...")
            freeze_intervals = detect_freeze_frames(working_video_path, min_duration=freeze_duration, noise_tolerance=freeze_noise)
            
        all_remove_intervals = silence_intervals + filler_intervals + freeze_intervals
        merged_remove_intervals = merge_intervals(all_remove_intervals)
        
        import subprocess
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", "-of", "json", working_video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        total_duration = float(probe_data['format']['duration'])
        
        keep_intervals = invert_intervals(merged_remove_intervals, total_duration)
        
        if len(keep_intervals) == 1 and keep_intervals[0] == (0.0, total_duration):
            logging.info("✅ No cuts needed!")
            return {'status': 'skipped', 'transcript': transcript_text}
            
        segment_files = extract_segments_ffmpeg(working_video_path, keep_intervals, temp_dir=".", target_bitrate="4M", progress_callback=progress_callback)
        codec, preset_value, ffmpeg_params = get_encoding_params(use_gpu_encoding, use_crf, bitrate, crf, preset)
        
        concatenate_segments_ffmpeg(segment_files, output_path, codec='copy' if no_crossfade else codec, preset=None if no_crossfade else preset_value, ffmpeg_params=[] if no_crossfade else ffmpeg_params, bitrate=None)
        
        return {'status': 'complete', 'output_path': output_path, 'transcript': transcript_text}
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {'status': 'error', 'error': str(e)}
        
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if transposed and os.path.exists(working_video_path):
            logging.info(f"Cleaning up transposed video: {working_video_path}")
            os.remove(working_video_path)
        logging.info("Done.")

if __name__ == "__main__":
    input_test_file = r"c:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-09 11.10.28 Lekh Poudel's Zoom Meeting\video1093907327.mp4"
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
            bitrate="5000k",
            preset="medium",
            use_crf=False,
            crf=18,
            filler_words=["um", "uh", "umm", "uhh", "er", "just, you know", "like, you know", "लाइग", "अनि", "मतलब", "हैत", "अ"],
            freeze_duration=5.0,
            remove_background=False,
            bg_color="green",
            bg_image=None,
            rvm_model="mobilenetv3",
            rvm_erode=0,
            rvm_dilate=0,
            rvm_median=0,
            rvm_blur=0,
            use_segmentation=False,
            seg_model="general",
            seg_threshold=0.5,
            seg_smooth=5
        )
    else:
        print(f"ℹ️ Test video path not found.")