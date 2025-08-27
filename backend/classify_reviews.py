import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# === Load Dataset ===
df = pd.read_csv("../data/cleaned_scraped_reviews.csv")

# Ensure the dataset has 'text' and 'category' columns
if "category" not in df.columns:
    raise ValueError("The dataset must have a 'category' column for fine-tuning.")

# === Convert to Hugging Face Dataset ===
dataset = Dataset.from_pandas(df)

# === Tokenize Data ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === Load Pre-trained Model ===
labels = df["category"].unique()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["category"].map(label2id)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(labels), id2label=id2label, label2id=label2id
)

# === Define Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# === Define Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# === Train the Model ===
trainer.train()

# === Save the Fine-Tuned Model ===
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("✅ Fine-tuning complete. Model saved to ./fine_tuned_model")
