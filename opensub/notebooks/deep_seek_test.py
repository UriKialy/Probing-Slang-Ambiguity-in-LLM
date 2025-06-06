import pandas as pd
import requests
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\sheff\PycharmProjects\Probing-Slang-Ambiguity-in-LLM\opensub\data\slang_OpenSub_filtered.tsv"
API_KEY = "sk-0c1a068b60c94111b1ea11285eeceb51"  # Replace with your actual key
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH, sep='\t')  # explicitly tell pandas it's tab-separated
df = df[['SENTENCE', 'FULL_CONTEXT', 'SLANG_TERM', 'ANNOTATOR_CONFIDENCE']].copy()
df.columns = ['sentence', 'context', 'slang', 'confidence']  # rename for consistency


# === API HEADERS ===
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# === FUNCTION: DeepSeek Query ===
def ask_deepseek_is_slang(slang_term, context):
    prompt = f"In the following context, is the word '{slang_term}' used as a slang expression? Reply with only 'Yes' or 'No'.\n\nContext:\n{context}"

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
print("\nQuerying DeepSeek for slang detection...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    is_slang = ask_deepseek_is_slang(row['slang'], row['context'])
    results.append(is_slang)
    time.sleep(1)  # avoid rate limits, adjust if needed

df['deepseek_slang'] = results
df = df[df['deepseek_slang'].notnull()]

# === METRICS: Accuracy Calculation ===
df['weight'] = df['confidence'].astype(int)
df['deepseek_binary'] = df['deepseek_slang'].astype(int)
df['correct'] = df['deepseek_binary']  # everything in dataset is slang

# Weighted accuracy
total_weight = df['weight'].sum()
correct_weight = (df['correct'] * df['weight']).sum()
accuracy = correct_weight / total_weight

print(f"\n=== ACCURACY REPORT ===")
print(f"Total samples: {len(df)}")
print(f"Weighted accuracy: {accuracy:.3f}")

# === PER CONFIDENCE LEVEL ===
print("\nAccuracy by confidence level:")
for level in sorted(df['confidence'].unique()):
    sub_df = df[df['confidence'] == level]
    w_acc = (sub_df['correct'] * sub_df['weight']).sum() / sub_df['weight'].sum()
    print(f"  Confidence {level}: {w_acc:.3f}")

# === OPTIONAL: Plot ===
try:
    df['conf_label'] = df['confidence'].map({1: 'Low', 2: 'Medium', 3: 'High'})
    grouped = df.groupby('conf_label').apply(lambda x: (x['correct'] * x['weight']).sum() / x['weight'].sum())
    grouped.plot(kind='bar', title='Accuracy by Annotator Confidence Level', ylabel='Accuracy')
    plt.tight_layout()
    plt.savefig("accuracy_by_confidence.png")
    print("\nPlot saved to 'accuracy_by_confidence.png'")
except Exception as e:
    print(f"Plotting failed: {e}")





