import pandas as pd

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\sheff\PycharmProjects\Probing-Slang-Ambiguity-in-LLM\opensub\data\slang_OpenSub_filtered.tsv"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH, sep='\t', usecols=["ANNOTATOR_CONFIDENCE"])

# === COUNT BY CONFIDENCE LEVEL ===
counts = df["ANNOTATOR_CONFIDENCE"].value_counts().sort_index()

# Ensure levels 1, 2, 3 are shown even if one is missing
for level in [1, 2, 3]:
    print(f"Confidence {level}: {counts.get(level, 0)}")
