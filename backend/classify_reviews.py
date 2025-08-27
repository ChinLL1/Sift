import pandas as pd
from transformers import pipeline

# === Load CSV ===
df = pd.read_csv("../data/cleaned_scraped_reviews.csv")

# Ensure you have a column with review text
reviews = df["text"].dropna().astype(str).tolist()

# === Define candidate labels (policy categories) ===
labels = [
    "relevant experience",
    "advertisement or promotion",
    "off-topic or irrelevant",
    "rant without actual visit evidence"
]

# === Load zero-shot classification pipeline ===
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Run classification ===
preds = []
for text in reviews:
    result = classifier(text, candidate_labels=labels, multi_label=False)
    preds.append(result["labels"][0])  # take top predicted label

# === Add predictions to dataframe ===
df["predicted_category"] = preds

# === Save to new CSV ===
df.to_csv("classified_reviews.csv", index=False)

print("✅ Classification complete. Results saved to classified_reviews.csv")
