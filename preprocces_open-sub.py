import pandas as pd
import re
from datasets import load_dataset

# Slang word list
slang_words = {"bad", "bang", "beat", "bet", "blow", "bomb", "booked", "bounce", "bread", "broke", "burn", "buzz",
               "calm", "cap", "catch", "check", "chef", "chill", "clap", "clean", "clutch", "cold", "come", "cook",
               "cool", "crack", "cringe", "cut", "dank", "dark", "dead", "deadass", "dope", "drag", "draw", "drip",
               "drop", "dust", "extra", "fam", "fire", "fit", "flex", "gas", "ghost", "glow", "grind", "grub", "hard",
               "hater", "head", "hit", "hot", "jam", "kick", "kill", "light", "link", "lit", "live", "loaded", "long",
               "loop", "loud", "lowkey", "mad", "man", "mood", "move", "off", "peak", "pop", "press", "pressed", "pull",
               "quiet", "ride", "ripped", "roll", "run", "safe", "salty", "savage", "secure", "serve", "shade", "shook",
               "sick", "slaps", "slay", "slide", "smoke", "snap", "snack", "soft", "spill", "squad", "stack", "stale",
               "stan", "stick", "sus", "swag", "tea", "thick", "thin", "thirsty", "tight", "ting", "tool", "touch",
               "trash", "trip", "turnt", "vibe", "wave", "wet", "whip", "woke", "work", "bag", "bars", "base", "brick",
               "cake", "cheese", "dash", "dip", "fade", "game", "heat", "ice", "juice", "plug", "poppin", "rack",
               "sauce", "score", "shine", "trap"}


def preprocess_sentence(sentence):
    """Clean and normalize a sentence from raw subtitle data."""
    if not isinstance(sentence, str):
        return ""
    # Remove timestamps (e.g., 00:01:23.456 --> 00:01:25.789)
    sentence = re.sub(r'\d+:\d+:\d+\.\d+ --> \d+:\d+:\d+\.\d+', '', sentence)
    # Remove subtitle IDs (e.g., numbers at the start of lines)
    sentence = re.sub(r'^\d+\s*', '', sentence)
    # Remove brackets, parentheses, and other metadata
    sentence = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', sentence)
    # Replace dashes with spaces and remove punctuation
    sentence = re.sub(r'[-–—]', ' ', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Normalize to lowercase and remove extra whitespace
    return sentence.lower().strip()


def slang_in_sentence(sentence):
    """Check if a sentence contains any slang word."""
    if not isinstance(sentence, str) or not sentence:
        return False
    tokens = set(sentence.split())
    return not slang_words.isdisjoint(tokens)


def process_opensubtitles(output_file="filtered_OpenSub_slang_dataset.tsv", sample_size=10000):
    """Extract sentences from OpenSubtitles, filter for slang, and save to TSV."""
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("open_subtitles", lang1="en")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure 'datasets' library is installed: pip install datasets")
        print("Check your internet connection or Hugging Face availability.")
        return

    sentences = []

    # Step 1: Extract and clean sentences
    # Access the 'train' split (or other splits if available)
    for item in dataset['train']:  # Adjust split name if necessary
        sentence = item['translation']['en']  # Assuming 'en' field for English sentences
        cleaned = preprocess_sentence(sentence)
        if cleaned:  # Only keep non-empty sentences
            sentences.append(cleaned)

    print(f"Extracted {len(sentences)} sentences from OpenSubtitles dataset")

    # Create DataFrame
    df = pd.DataFrame(sentences, columns=['SENTENCE'])

    # Step 2: Filter sentences containing slang
    filtered_df = df[df['SENTENCE'].apply(slang_in_sentence)]

    # Limit to sample_size to avoid overly large output
    if len(filtered_df) > sample_size:
        filtered_df = filtered_df.sample(n=sample_size, random_state=42)

    # Step 3: Save filtered dataset
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved {len(filtered_df)} slang-containing sentences to {output_file}")


# Main execution
process_opensubtitles()