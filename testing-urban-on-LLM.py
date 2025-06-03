import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load filtered dataset
df = pd.read_csv("filtered_urban_dictionary.csv")

# Models to test
MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-llm-7b-base"  # adjust if needed
}

# You can change task from sentiment-analysis to another depending on use case
task = "text-classification"

# Load models/pipelines
model_pipelines = {}
for name, model_id in MODELS.items():
    print(f"Loading {name}...")
    pipe = pipeline(task, model=model_id, tokenizer=model_id, device=0 if torch.cuda.is_available() else -1)
    model_pipelines[name] = pipe

# Process a few definitions
N = 10  # Adjust for number of rows to evaluate
results = []

print("\nRunning inference...\n")
for i in range(min(N, len(df))):
    row = df.iloc[i]
    definition = row['definition']
    example = row['example']
    text = f"{definition} Example: {example}"

    row_result = {"index": i, "word": row['word'], "text": text}
    for model_name, pipe in model_pipelines.items():
        try:
            output = pipe(text, truncation=True)
            row_result[model_name] = output
        except Exception as e:
            row_result[model_name] = f"Error: {str(e)}"

    results.append(row_result)

# Display results
for r in results:
    print(f"\n--- Entry #{r['index']} ---")
    print(f"Word: {r['word']}")
    print(f"Text: {r['text']}")
    for model_name in MODELS:
        print(f"{model_name}: {r[model_name]}")
