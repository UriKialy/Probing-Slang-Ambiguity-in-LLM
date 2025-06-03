import re
from datasets import load_dataset

slang_words = [
    "bad", "bang", "beat", "bet", "blow", "bomb", "booked", "bounce", "bread", "broke",
    "burn", "buzz", "calm", "cap", "catch", "check", "chef", "chill", "clap", "clean",
    "clutch", "cold", "come", "cook", "cool", "crack", "cringe", "cut", "dank", "dark",
    "dead", "deadass", "dope", "drag", "draw", "drip", "drop", "dust", "extra", "fam",
    "fire", "fit", "flex", "gas", "ghost", "glow", "grind", "grub", "hard", "hater",
    "head", "hit", "hot", "jam", "kick", "kill", "light", "link", "lit", "live",
    "loaded", "long", "loop", "loud", "lowkey", "mad", "man", "mood", "move", "off",
    "peak", "pop", "press", "pressed", "pull", "quiet", "ride", "ripped", "roll", "run",
    "safe", "salty", "savage", "secure", "serve", "shade", "shook", "sick", "slaps",
    "slay", "slide", "smoke", "snap", "snack", "soft", "spill", "squad", "stack",
    "stale", "stan", "stick", "sus", "swag", "tea", "thick", "thin", "thirsty", "tight",
    "ting", "tool", "touch", "trash", "trip", "turnt", "vibe", "wave", "wet", "whip",
    "woke", "work", "bag", "bars", "base", "brick", "cake", "cheese", "dash", "dip",
    "fade", "game", "heat", "ice", "juice", "plug", "poppin", "rack", "sauce", "score",
    "shine", "trap"
]

slang_words_set = set(slang_words)

def contains_slang_word(example):
    # Combine all relevant fields and lowercase
    text = ' '.join([example['word'], example['definition'], example['example']]).lower()
    # Extract tokens: words only, ignoring punctuation
    tokens = re.findall(r'\b\w+\b', text)
    # Return True if any token matches slang words exactly
    return any(token in slang_words_set for token in tokens)

ds = load_dataset("daspartho/urban_dictionary")

filtered_ds = ds['train'].filter(contains_slang_word)

print(f"Original size: {len(ds['train'])}, Filtered size: {len(filtered_ds)}")

filtered_ds.to_csv("filtered_urban_dictionary.csv", index=False)
