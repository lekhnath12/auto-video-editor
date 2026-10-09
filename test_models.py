import os
import time
from transformers import pipeline

def test_nepali_models():
    audio_path = r"C:\Users\lekhp\OneDrive\Desktop\temp_audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at: {audio_path}")
        return

    # List of applicable Nepali and multilingual Whisper models to test
    models_to_test = [
       "aamoshdahal/whisper-small-nepali",
        "sumanpaudel1997/nepali-asr-whisper-turbo"
    ]

    print(f"=== Starting Model Comparison for: {audio_path} ===\n")

    for model_id in models_to_test:
        print(f"--------------------------------------------------")
        print(f"🔄 Loading and testing model: {model_id}")
        print(f"--------------------------------------------------")
        
        start_time = time.time()
        try:
            # Initialize Hugging Face ASR pipeline on CPU (device=-1)
            asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=-1
            )

            # Transcribe audio using chunked pipeline with Nepali language constraint
            result = asr_pipe(
                audio_path,
                chunk_length_s=30,
                return_timestamps=True,
                generate_kwargs={"language": "ne", "task": "transcribe"}
            )

            transcript = result.get("text", "").strip()
            elapsed_time = time.time() - start_time

            print(f"✅ SUCCESS ({elapsed_time:.2f}s)")
            print(f"📝 Transcript:\n{transcript}\n")

        except Exception as e:
            print(f"❌ FAILED with model {model_id}: {e}\n")

    print("=== All model tests completed ===")

if __name__ == "__main__":
    test_nepali_models()