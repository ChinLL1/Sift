import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset

# === Load Dataset ===
df = pd.read_csv("../data/cleaned_scraped_reviews.csv")

# Debug: Print dataset columns
print("Dataset columns:", df.columns)

# Ensure the dataset has the correct columns
if "text" not in df.columns or "category" not in df.columns:
    raise ValueError("The dataset must have 'text' and 'category' columns for fine-tuning.")

# Map categories to integers
labels = df["category"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["category"].map(label2id)

# Debug: Print label mapping
print("Label mapping:", label2id)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Print statements to check the label mapping
print(df["label"].head())  # Check the first few labels
print(df["label"].dtype)  # Ensure the labels are integers

# === Tokenize Data ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,  # Ensure inputs are truncated to the model's max length
        padding="max_length",  # Pad inputs to the model's max length
        max_length=512,  # Set a maximum length (adjust based on your model)
        return_tensors="pt",  # Return PyTorch tensors
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === Load Pre-trained Model ===
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=len(labels), id2label=id2label, label2id=label2id
)

# === Define Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    logging_strategy="epoch",  # Replace 'evaluation_strategy' with 'logging_strategy'
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# === Define Data Collator ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Define Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Add the data collator here
)

# === Train the Model ===
trainer.train()

# === Save the Fine-Tuned Model ===
model.save_pretrained("C:/Users/yapor/OneDrive/Documents/GitHub/Sift/backend/fine_tuned_deberta")
tokenizer.save_pretrained("C:/Users/yapor/OneDrive/Documents/GitHub/Sift/backend/fine_tuned_deberta")

print("✅ Fine-tuning complete. Model saved to ./fine_tuned_deberta")
