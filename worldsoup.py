import os
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_windows_font_path():
    """Finds a valid Windows font supporting Devanagari text."""
    fonts_dir = r"C:\Windows\Fonts"
    
    # Priority list of Windows fonts with Devanagari support
    candidates = [
        "mangal.ttf",
        "mangal.TTF",
        "Nirmala.ttf",
        "Nirmala.TTF",
        "aparaj.ttf",
        "kokila.ttf"
    ]
    
    for font in candidates:
        full_path = os.path.join(fonts_dir, font)
        if os.path.exists(full_path):
            return full_path
            
    # Fallback search if exact file names differ
    if os.path.exists(fonts_dir):
        for f in os.listdir(fonts_dir):
            if f.lower().startswith(("mangal", "nirmala", "aparaj")):
                return os.path.join(fonts_dir, f)
                
    raise FileNotFoundError("Could not find a valid Devanagari-compatible font in C:\\Windows\\Fonts")

def generate_devanagari_word_soup(transcript_path: str, output_image_path: str = "word_soup.png"):
    if not os.path.exists(transcript_path):
        print(f"❌ Transcript file not found: {transcript_path}")
        return

    print(f"📖 Reading transcript from: {transcript_path}")
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Extract only Devanagari words using Unicode ranges (U+0900 to U+097F)
    devanagari_words = re.findall(r'[\u0900-\u097F]+', text)

    if not devanagari_words:
        print("⚠️ No Devanagari words found in the transcript.")
        return

    # Filter out common stopwords
    stopwords = {"र", "छ", "को", "मा", "कि", "पनि", "यो", "त्यो", "भने", "हुन्छ", "भयो", "र", "छ"}
    filtered_words = [word for word in devanagari_words if word not in stopwords and len(word) > 1]

    # 2. Count Frequencies
    word_counts = Counter(filtered_words)
    
    print("\n📊 Top 20 Most Frequent Devanagari Words:")
    for word, freq in word_counts.most_common(20):
        print(f"  {word}: {freq}")

    # 3. Get valid font path dynamically
    font_path = get_windows_font_path()
    print(f"\n🎨 Using font path: {font_path}")
    
    # 4. Generate Word Cloud Image
    wordcloud = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        max_words=150,
        colormap="viridis"
    ).generate_from_frequencies(word_counts)

    # 5. Save and Display
    plt.figure(figsize=(16, 9), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    plt.savefig(output_image_path, dpi=300)
    print(f"✅ Word soup image successfully saved to: {output_image_path}")
    
    plt.show()

if __name__ == "__main__":
    transcript_file = r"C:\Users\lekhp\OneDrive\Documents\Zoom\2026-10-02 21.20.58 Interview With Dr. Khadka\video1184682387_temp_audio_transcript.txt"
    output_image = "dr_khadka_interview_word_soup.png"
    
    generate_devanagari_word_soup(transcript_file, output_image)