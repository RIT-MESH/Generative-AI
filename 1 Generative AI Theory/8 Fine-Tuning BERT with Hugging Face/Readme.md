# Fine-Tuning BERT with Hugging Face in README.md



Below is a complete guide  to fine-tune a pre-trained BERT model for text classification using the Hugging Face `transformers` library. This example uses the IMDB dataset for sentiment analysis.

---


https://github.com/user-attachments/assets/aa5166d2-ee87-4d9c-8c7b-309f38cbdd7b

---
## Prerequisites

- **Libraries**: Install the required packages:
  ```bash
  pip install transformers datasets torch
  ```
- **Task**: Text classification (sentiment analysis).
- **Dataset**: IMDB dataset from Hugging Face `datasets`.

---

## Step-by-Step Guide

### 1. Load Pre-trained BERT Model and Tokenizer

Load the BERT tokenizer and model for sequence classification.

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Specify the model name
model_name = "bert-base-uncased"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the model with a classification head (2 labels for binary classification)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

---

### 2. Prepare the Dataset

Load the IMDB dataset and tokenize it.

```python
from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Define the tokenization function
# Applies padding, truncation, and sets max sequence length to 512 tokens
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the entire dataset in batches
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Shuffle and select a small subset for training and evaluation to reduce training time
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
```

---

### 3. Convert Dataset to PyTorch Format

Set the dataset format to PyTorch-compatible inputs.

```python
# Rename 'label' column to 'labels' as required by Hugging Face models
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Convert to PyTorch format
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Set formats for training and evaluation datasets
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

---

### 4. Set Up DataLoader

Create PyTorch `DataLoader` objects for batching.

```python
from torch.utils.data import DataLoader

# DataLoader for training with shuffling
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

# DataLoader for evaluation without shuffling
eval_dataloader = DataLoader(eval_dataset, batch_size=8)
```

---

### 5. Define Optimizer and Scheduler

Configure the optimizer and learning rate scheduler.

```python
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Calculate total training steps (used for scheduling learning rate)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# Create a linear learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

### 6. Training Loop

Train the model over multiple epochs.

```python
from tqdm import tqdm  # Progress bar

model.train()  # Set model to training mode
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_dataloader):
        # Move batch data to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()  # Clear gradients

    print(f"Loss: {loss.item()}")
```

---

### 7. Evaluation

Evaluate the model’s performance on the validation set.

```python
from sklearn.metrics import accuracy_score

model.eval()  # Set model to evaluation mode
predictions, true_labels = [], []

# Disable gradient computation for evaluation
with torch.no_grad():
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        # Get predicted class
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(batch["labels"].cpu().tolist())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy:.4f}")
```

---

### 8. Save the Fine-Tuned Model

Save the trained model and tokenizer.

```python
# Save model weights
model.save_pretrained("fine_tuned_bert_imdb")

# Save tokenizer files
tokenizer.save_pretrained("fine_tuned_bert_imdb")
```

---

### 9. Inference (Optional)

Perform inference with the fine-tuned model.

```python
from transformers import pipeline

# Load the fine-tuned model and tokenizer
classifier = pipeline("sentiment-analysis", model="fine_tuned_bert_imdb", tokenizer="fine_tuned_bert_imdb")

# Run prediction
result = classifier("This movie is great!")
print(result)  # Example: [{'label': 'LABEL_1', 'score': 0.95}]
```

---

## Using the Trainer API (Alternative)

For a simpler approach, use the `Trainer` API.

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Evaluate the trained model
trainer.evaluate()
```

---

## Key Considerations

- **Hardware**: BERT-base requires ~6-8GB VRAM for a batch size of 8.
- **Hyperparameters**:
  - Learning rate: 1e-5 to 5e-5.
  - Epochs: 2-4.
  - Batch size: 4-32.
- **Task-Specific Heads**: Use `BertForTokenClassification` for NER, etc.

---

