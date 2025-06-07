import pandas as pd
import requests
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\sheff\PycharmProjects\Probing-Slang-Ambiguity-in-LLM\opensub\data\slang_OpenSub_filtered.tsv"
API_KEY = "sk-0c1a068b60c94111b1ea11285eeceb51"  # Make sure there are no extra spaces
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH, sep='\t')
df = df[['SENTENCE', 'FULL_CONTEXT', 'SLANG_TERM', 'ANNOTATOR_CONFIDENCE']].copy()
df.columns = ['sentence', 'context', 'slang', 'confidence']

# --- FILTER TO ONLY CONFIDENCE == 3 ---
df = df[df['confidence'] == 3].reset_index(drop=True)

# === API HEADERS ===
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# === FUNCTION: DeepSeek Query ===
def ask_deepseek_is_slang(slang_term, context):
    prompt = (
        f"In the following context, is the word '{slang_term}' used as a slang expression? "
        "Reply with only 'Yes' or 'No'.\n\n"
        f"Context:\n{context}"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()['choices'][0]['message']['content'].strip().lower()
        return 'yes' in reply
    except Exception as e:
        print(f"API Error: {e}")
        return None

# === APPLY API TO EACH ROW ===
results = []
print("\nQuerying DeepSeek for slang detection (only confidence=3)...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    is_slang = ask_deepseek_is_slang(row['slang'], row['context'])
    results.append(is_slang)
    time.sleep(1)  # to avoid hammering the API

df['deepseek_slang'] = results
df = df[df['deepseek_slang'].notnull()]  # drop any rows where the API call failed

# === METRICS: Accuracy Calculation ===
df['weight'] = df['confidence'].astype(int)       # all are 3, but keep it general
df['deepseek_binary'] = df['deepseek_slang'].astype(int)
df['correct'] = df['deepseek_binary']             # since every example is truly slang

# Weighted accuracy
total_weight = df['weight'].sum()
correct_weight = (df['correct'] * df['weight']).sum()
accuracy = correct_weight / total_weight if total_weight > 0 else 0.0

print(f"\n=== ACCURACY REPORT (confidence=3 only) ===")
print(f"Total samples: {len(df)}")
print(f"Weighted accuracy: {accuracy:.3f}")

# Accuracy by confidence (trivially all 3)
print("\nAccuracy by confidence level (should show only level 3):")
for level in sorted(df['confidence'].unique()):
    sub_df = df[df['confidence'] == level]
    w_acc = (sub_df['correct'] * sub_df['weight']).sum() / sub_df['weight'].sum()
    print(f"  Confidence {level}: {w_acc:.3f}")

# === SAVE MISCLASSIFIED ROWS TO A SEPARATE FILE ===
# Misclassified = deepseek_binary == 0 (DeepSeek said "No" but the row is truly slang)
mis_df = df[df['deepseek_binary'] == 0].copy()
if not mis_df.empty:
    mis_df.to_csv("misclassified_confidence3.csv", index=False)
    print(f"\nSaved {len(mis_df)} misclassified rows to 'misclassified_confidence3.csv'")
else:
    print("\nNo misclassifications found—no file written.")

# === OPTIONAL: Plot ===
try:
    df['conf_label'] = df['confidence'].map({1: 'Low', 2: 'Medium', 3: 'High'})
    grouped = df.groupby('conf_label').apply(
        lambda x: (x['correct'] * x['weight']).sum() / x['weight'].sum()
    )
    grouped.plot(
        kind='bar',
        title='Accuracy by Annotator Confidence Level (High only)',
        ylabel='Accuracy'
    )
    plt.tight_layout()
    plt.savefig("accuracy_by_confidence.png")
    print("Plot saved to 'accuracy_by_confidence.png'")
except Exception as e:
    print(f"Plotting failed: {e}")
