# Learning Architectures: RNN, CNN, and Transformers

## Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data. They are widely used in **text processing, speech recognition, and time-series analysis**, as they can maintain contextual information.

---

## 📌 **Types of RNNs**
RNNs come in different architectures to improve efficiency and accuracy:

### 🔹 **Stacked RNNs**
Stacked RNNs consist of multiple RNN layers stacked on top of each other to **learn complex representations** from sequential data.

### 🔹 **Bidirectional RNNs**
Bidirectional RNNs process input in both forward and backward directions, helping capture **context from both past and future words**.

### 🔹 **Sequence-to-Sequence Models**
Sequence-to-sequence models **convert one sequence into another** and can handle both **fixed and variable-length sequences**.

---

## 📌 **Sequence to Sequence Learning**
RNNs can be categorized based on input/output structure:

👉 **Many-to-One** → Used for **text classification** (e.g., sentiment analysis, spam detection).  
👉 **One-to-Many** → Applied in **image captioning**.  
👉 **Many-to-Many** → Used in **POS tagging, Named Entity Recognition (NER), text summarization, machine translation, and question answering**.

---

## 📌 **How RNNs Work**
RNNs process sequences **one element at a time**, maintaining a **hidden state** that stores relevant information.

### ✨ **Example of RNN Processing**
Let's consider an example sentence:
> **"I love AI"**  
The hidden state updates as each word is processed.

⚠️ **Limitation**: Standard RNNs struggle with **long-term dependencies**, leading to the **vanishing gradient problem**.

---

## 📌 **Long Short-Term Memory (LSTM)**
LSTMs solve the **vanishing gradient problem** by introducing a **memory cell** that allows for better retention of long-term dependencies.

### 🔹 **Key Features of LSTM**
- **Memory Cell**: Maintains long-term dependencies.
- **Gates**:
  - **Forget Gate** → Decides what information to discard.
  - **Input Gate** → Decides what new information to store.
  - **Output Gate** → Controls the output based on memory state.

### 📝 **Example: LSTM in Python**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create an LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(100, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Display model summary
model.summary()
```

---

## 📌 **Gated Recurrent Unit (GRU)**
GRUs are a simpler alternative to LSTMs with fewer parameters, making them **faster to train**.

### 🔹 **LSTM vs GRU**
| Feature   | LSTM  | GRU  |
|-----------|-------|------|
| Complexity | Higher | Lower |
| Training Time | Slower | Faster |
| Performance | Better for long sequences | Faster with fewer parameters |

### 📝 **Example: GRU in Python**
```python
from tensorflow.keras.layers import GRU

# Create a GRU model
model = Sequential([
    GRU(50, activation='relu', input_shape=(100, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```

---

## 📌 **Transformers: The Evolution Beyond RNNs**
Transformers **do not rely on recurrence** but instead use **self-attention mechanisms** to process sequences in parallel.

### 🔹 **Key Features of Transformers**
- **Self-Attention** → Determines importance of different words in a sequence.
- **Parallel Processing** → Speeds up training.
- **Better Long-Term Dependency Handling** → Overcomes RNN limitations.

### 📝 **Example: Transformer Model in Python**
```python
from transformers import AutoModel, AutoTokenizer

# Load a pre-trained transformer model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize a sample sentence
text = "Transformers are powerful models for NLP tasks."
inputs = tokenizer(text, return_tensors="pt")

# Get model output
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

---

## 📌 **Conclusion**
RNNs, LSTMs, and GRUs are powerful for sequential data processing, but **transformers have revolutionized the field** with their efficiency and accuracy.

👉 RNNs → **Good for simple sequences**  
👉 LSTMs/GRUs → **Better for long-term dependencies**  
👉 Transformers → **State-of-the-art for NLP and sequence tasks**  

🔗 **Want to learn more?** Explore the code, contribute, or open an issue!

---

