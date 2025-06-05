import pandas as pd
import re

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

pattern = re.compile(r"\b(" + "|".join(re.escape(word) for word in slang_words) + r")\b", flags=re.IGNORECASE)

# Filter positive (slang) file
df_pos = pd.read_csv("../data/slang_OpenSub.tsv", sep="\t", dtype=str)
df_pos_filtered = df_pos[df_pos["SENTENCE"].str.contains(pattern, na=False)]
df_pos_filtered.to_csv("slang_OpenSub_filtered.tsv", sep="\t", index=False)

# Filter negative (non-slang) file
df_neg = pd.read_csv("../data/slang_OpenSub_negatives.tsv", sep="\t", dtype=str)
df_neg_filtered = df_neg[df_neg["SENTENCE"].str.contains(pattern, na=False)]
df_neg_filtered.to_csv("slang_OpenSub_negatives_filtered.tsv", sep="\t", index=False)
