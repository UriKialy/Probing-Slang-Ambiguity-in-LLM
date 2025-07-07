import pandas as pd
import requests
from tqdm import tqdm
import time
from keys_do_not_upload import  deepseek_API_key
# === CONFIGURATION ===
NEG_DATA_PATH = r"C:\Users\sheff\PycharmProjects\Probing-Slang-Ambiguity-in-LLM\opensub\data\slang_OpenSub_negatives_filtered.tsv"
API_KEY       = deepseek_API_key
API_URL       = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME    = "deepseek-chat"

# === LOAD NEGATIVE DATA ===
# We only care about the 'SENTENCE' column here
neg_df = pd.read_csv(NEG_DATA_PATH, sep='\t', usecols=["SENTENCE"]).copy()
neg_df.columns = ["sentence"]  # rename for consistency

# === API HEADERS ===
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# === FUNCTION: Ask DeepSeek if a sentence contains slang ===
def ask_deepseek_sentence_has_slang(sentence: str) -> bool | None:
    """
    Returns True if DeepSeek replies 'Yes' (i.e. it thinks the sentence has slang),
    False if it replies 'No', or None if there's an API error.
    """
    prompt = (
        "Does the following sentence contain any slang expression? "
        "Reply with only 'Yes' or 'No'.\n\n"
        f"Sentence:\n{sentence}"
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

# === APPLY API TO EACH ROW (NEGATIVES) ===
results = []
print("\nQuerying DeepSeek on negative sentences...")
for _, row in tqdm(neg_df.iterrows(), total=len(neg_df)):
    guess = ask_deepseek_sentence_has_slang(row['sentence'])
    results.append(guess)
    time.sleep(1)  # respect rate limits; adjust if you have a higher quota

neg_df['deepseek_thinks_slang'] = results

# Drop any rows where the API call returned None (error)
neg_df = neg_df[neg_df['deepseek_thinks_slang'].notnull()].reset_index(drop=True)

# === COMPUTE ACCURACY ON NEGATIVE SET ===
# A “correct” negative is when DeepSeek says False (it did NOT detect slang)
total_neg = len(neg_df)
correct_neg = (neg_df['deepseek_thinks_slang'] == False).sum()
accuracy_neg = correct_neg / total_neg if total_neg > 0 else 0.0

print(f"\n=== NEGATIVE-SET ACCURACY ===")
print(f"Total negative samples evaluated: {total_neg}")
print(f"Number correctly classified as non-slang: {correct_neg}")
print(f"Accuracy on negatives: {accuracy_neg:.3f}")

# === SAVE ACCURACY TO CSV ===
acc_df = pd.DataFrame({"negative_accuracy": [accuracy_neg]})
acc_df.to_csv("accuracy_negatives.csv", index=False)
print("Saved negative-set accuracy to 'accuracy_negatives.csv'")

# === SAVE MISCLASSIFIED NEGATIVES ===
# Misclassified = DeepSeek said True (i.e. it detected slang) but these are negatives
mis_df = neg_df[neg_df['deepseek_thinks_slang'] == True].copy()
if not mis_df.empty:
    mis_df.to_csv("misclassified_negatives.csv", index=False)
    print(f"Saved {len(mis_df)} false-positives to 'misclassified_negatives.csv'")
else:
    print("No false-positives found—no file written for misclassifications.")
