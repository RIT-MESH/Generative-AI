### Transformer Architecture, Contextual Embedding, and Self-Attention 

## **Transformer Architecture**

### **Theory**

#### **Transformer Summary**
The **transformer model** is a deep learning architecture designed to handle sequential data more efficiently than traditional recurrent neural networks (RNNs). It is widely used in NLP tasks, including machine translation, text summarization, and language modeling.

The **transformer model** uses an **encoder** and a **decoder**.

- **Encoder**:
  - Steps: **Input embedding, positional encoding, self-attention, multi-head attention, add and norm layer, feed-forward network**.
  - The original research paper uses six identical blocks.
- **Decoder**:
  - Steps: **Output sequence embedding, positional encoding, masked multi-head attention, add and norm, residual connection, multi-head attention, linear activation function, softmax layer**.

#### **Positional Encoding**
Transformers process inputs in parallel, so positional encoding ensures the model retains word order.

- Uses sine and cosine functions to encode position.
- *Example*: "I love you" – positional encoding ensures the model understands word order.

The formula for positional encoding involves sine and cosine functions.

#### **Encoder Details**
- **Tokenization**: Input text is converted into tokens.
- **Word Embeddings**: Each token is transformed into a dense vector representation.
- **Positional Encoding**: Provides information about token positions.
- **Multi-headed self-attention**: Computes relationships between tokens.
- **Feed-forward network**: Applies transformations to enhance feature extraction.
- **Layer normalization and residual connections**: Improve training stability.

The encoder block is repeated N times.

**Residual Connection**: Also known as **skip connections**, these help stabilize training and prevent vanishing gradients.

*Example*: Imagine training a very deep neural network. As the network learns, the gradients (signals that update the network's weights) can become very small, making learning slow or impossible. Skip connections allow the original input to bypass some layers, ensuring the gradient signal can flow more easily through the network.

**Layers**: The use of **6 Encoders** and **6 Decoders** is a **hyperparameter**. This means it's a design choice that can be tuned to optimize performance.

---

### **Code Implementation with Dummy Inputs and Outputs**

#### **Transformer Model Implementation**
```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

    def forward(self, src, tgt):
        encoded_src = self.encoder(src)
        decoded_output = self.decoder(tgt, encoded_src)
        return decoded_output

# Example Usage
dummy_src = torch.rand(10, 32, 512)  # (sequence length, batch size, embedding size)
dummy_tgt = torch.rand(10, 32, 512)
model = TransformerModel(embed_size=512, heads=8, forward_expansion=4, dropout=0.1)
output = model(dummy_src, dummy_tgt)
print("Transformer Model Output:")
print(output.shape)
```
**Output:**
```
torch.Size([10, 32, 512])
```

## **Contextual Embedding and Self-Attention**

### **Theory**

#### **Word Embedding Models**
Word embeddings convert words into dense vector representations, allowing models to capture semantic relationships between words.

- Traditional embeddings (e.g., Word2Vec, GloVe) create **static** representations.
- Contextual embeddings (e.g., BERT, GPT) generate **dynamic** representations that vary based on surrounding words.

*Example*: Word2Vec is a model that maps words to vectors, capturing semantic similarities. Words like "king" and "queen" will have vectors closer to each other than "king" and "bicycle".

#### **Contextual Embedding**
- Self-attention enables contextual embedding, making word representations dynamic.
- *Example*: "Apple" in "An apple a day keeps the doctor away" vs. "Apple makes great phones" – embedding differs based on context.

#### **Self-Attention Mechanism**
The **self-attention** mechanism allows a model to weigh the importance of different words in a sequence while processing a specific word.

- **Query (Q), Key (K), and Value (V)** vectors are computed for each word.
- **Similarity scores** are calculated between Q and K.
- **Softmax function** normalizes these scores.
- **Weighted sum of values (V)** forms the final representation.

This mechanism enables models to **capture long-range dependencies** more effectively than RNNs.

**Transformer**: Transformers use self-attention.

#### **Connections and Concepts**
- **NLP Pipeline**: Data ingestion, preprocessing, model training, evaluation.
- **Transfer Learning**: Transformers enable pre-training and fine-tuning.
- **Word Embedding**: Transformers build upon word embedding techniques, utilizing contextual and dynamic embeddings through self-attention.
- **Language Models**: Transformers enhance language models and enable text summarization.

---

### **Code Implementation with Dummy Inputs and Outputs**
#### **Contextual Embedding Implementation**
```python
import torch
import torch.nn as nn

class ContextualEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(ContextualEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

# Example usage
dummy_input = torch.randint(0, 1000, (5,))  # 5 random words from vocabulary
model = ContextualEmbedding(vocab_size=1000, embed_size=16)
output = model(dummy_input)
print("Contextual Embedding Output:")
print(output)
```
**Output:**
```
tensor([[ 0.1234,  0.5678, ...,  0.9821],
        [-0.3215,  0.6742, ..., -0.1534],
        ...])
```

#### **Self-Attention Mechanism Implementation**
```python
import torch
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        attention = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embed_size ** (1/2))
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, values)
        return self.fc_out(out)

# Example usage
dummy_input = torch.rand(10, 32, 64)  # (sequence length, batch size, embedding size)
attention = SelfAttention(embed_size=64, heads=8)
output = attention(dummy_input, dummy_input, dummy_input)
print("Self-Attention Output:")
print(output.shape)
```
**Output:**
```
torch.Size([10, 32, 64])
```

This document provides a structured overview of **Transformer architecture, contextual embedding, and self-attention** with **detailed theory explanations, Python implementations, dummy inputs, and output verification**. 

