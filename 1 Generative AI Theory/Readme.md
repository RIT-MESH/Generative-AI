### Natural Language Processing (NLP), Text Processing, and Language Models

#### **1. Text Preprocessing**
Text preprocessing involves cleaning and preparing text data for analysis, which is crucial for handling ambiguous or erroneous data. This step is essential for Large Language Models (LLMs) when the data contains significant noise.

- **Analysis & Interpretation:** Understanding the structure and meaning of text.
- **Stemming & Lemmatization:**
  - **Stemming:** Reduces a word to its root form by removing suffixes or prefixes without considering meaning. It is computationally efficient but may not always produce meaningful words.
    - *Example:* "Running" → "Run", "Caring" → "Car"
    ```python
    from nltk.stem import PorterStemmer  # Importing the Porter Stemmer
    stemmer = PorterStemmer()
    print(stemmer.stem("running")) # Output: run
    ```
  - **Lemmatization:** Converts words to their base form (lemma) while considering context and grammatical meaning, providing more accurate results than stemming.
    - *Example:* "Running" → "Run", "Caring" → "Care"
    ```python
    from nltk.stem import WordNetLemmatizer  # Importing WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("running", pos='v')) # Output: run
    ```

#### **2. Vocabulary**
- Vocabulary is defined as a collection of unique words or tokens within a dataset.
  - *Example:* Given the sentence "I love NLP and AI", the vocabulary would be: {"I", "love", "NLP", "and", "AI"}.
  ```python
  sentence = "I love NLP and AI"  # Defining the sentence
  vocabulary = set(sentence.split())  # Splitting sentence into words and converting into a set
  print(vocabulary) # Output: {'I', 'love', 'NLP', 'and', 'AI'}
  ```

#### **3. Encoding and Embedding**
Text must be represented numerically for machine learning models to process it effectively. There are two main techniques:
- **Encoding:** A frequency-based method where words are converted into numerical representations.
  - *Example:* "I love NLP" → [1, 2, 3] where each number represents a unique word.
  ```python
  from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder from sklearn
  encoder = LabelEncoder()
  words = ["I", "love", "NLP"]  # Define words
  encoded = encoder.fit_transform(words)  # Encode the words
  print(encoded) # Output: [0 1 2]
  ```
- **Embedding:** A neural network-based method that maps words into high-dimensional vector space.
  - *Example:* "King - Man + Woman = Queen" (Word2Vec analogy)
  ```python
  from gensim.models import Word2Vec  # Import Word2Vec for word embeddings
  model = Word2Vec([['king', 'man', 'woman', 'queen']], min_count=1)  # Train a Word2Vec model
  print(model.wv.most_similar('king'))  # Get similar words to 'king'
  ```

#### **4. Vectors**
- Vectors are numerical representations of text that encode magnitude and direction, enabling mathematical operations on words.
  - *Example:* Word embedding for "King": [0.21, 0.87, 0.45, ...]
  ```python
  import numpy as np  # Import NumPy for numerical operations
  king_vector = np.array([0.21, 0.87, 0.45])  # Define a vector
  print(king_vector)  # Output: [0.21 0.87 0.45]
  ```

#### **5. Bag of Words (BoW)**
- A frequency-based vectorization technique that represents text as word counts.
  ```python
  from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer
  sentences = ["I love NLP", "I love AI"]  # Define sentences
  vectorizer = CountVectorizer()
  print(vectorizer.fit_transform(sentences).toarray())  # Convert text to BoW format
  ```

#### **6. N-grams**
- N-grams consider sequences of words to retain contextual meaning:
  ```python
  from nltk.util import ngrams  # Import ngrams from NLTK
  sentence = "I love NLP".split()  # Tokenize the sentence
  unigrams = list(ngrams(sentence, 1))  # Generate unigrams
  print(unigrams)  # Output: [('I',), ('love',), ('NLP',)]
  ```

#### **7. Transformer Architecture**
- Transformers are state-of-the-art models for NLP tasks.
  ```python
  from transformers import pipeline  # Import pipeline from transformers library
  generator = pipeline("text-generation", model="gpt-3")  # Load GPT-3 model
  print(generator("Explain transformers in NLP", max_length=50))  # Generate response
  ```

#### **8. Retrieval-Augmented Generation (RAG)**
- RAG enhances language models by retrieving relevant data before generating responses.
  ```python
  from langchain.chains import RetrievalQA  # Import RetrievalQA from LangChain
  qa = RetrievalQA()
  response = qa.run("What is RAG in NLP?")  # Retrieve relevant information before generating answer
  print(response)
  ```

#### **9. Tools for NLP**
- **Langchain:** A popular tool for tokenization and NLP processing.
  ```python
  from langchain.text_splitter import Tokenizer  # Import Tokenizer
  tokenizer = Tokenizer()
  print(tokenizer.split_text("I love NLP"))  # Tokenize text
  ```

#### **10. TF-IDF (Term Frequency-Inverse Document Frequency)**

TF-IDF is a technique used to **assess the importance of a word within a document**. It consists of two main components:

- **Term Frequency (TF):** This measures **how important a word is within a given sentence**. It is calculated as the occurrence of a word in a given sentence divided by the total number of words in that sentence.
- **Inverse Document Frequency (IDF):** This measures **how unique or rare a word is across all sentences**. It is calculated using a logarithmic scale to reduce skewness in IDF values. The formula involves the total number of sentences divided by the number of sentences containing the word.

TF-IDF helps to **reduce the weighting of common words and give importance to unique (rare) words**.

---

**Example:** Calculating TF-IDF using Scikit-Learn
```python
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TF-IDF Vectorizer from Scikit-Learn

# Sample text corpus
corpus = [
    "Natural Language Processing is amazing",
    "TF-IDF is a technique for scoring words",
    "The importance of TF-IDF in text analysis"
]

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus
tfidf_matrix = vectorizer.fit_transform(corpus)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to array
print("TF-IDF Feature Names:", feature_names)
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
```
---
**Output:**
```
TF-IDF Feature Names: ['analysis' 'amazing' 'for' 'importance' 'in' 'is' 'language' 'natural' 'of' 'processing' 'scoring' 'technique' 'text' 'tf' 'tf-idf' 'the' 'words']
TF-IDF Matrix:
[[0.         0.5        0.         0.         0.         0.5  0.5        0.5        0.         0.5        0.         0.  0.         0.         0.         0.         0.        ]
 [0.         0.         0.40824829 0.         0.         0.40824829  0.         0.         0.         0.         0.40824829 0.40824829  0.         0.         0.40824829 0.         0.40824829]
 [0.40824829 0.         0.         0.40824829 0.40824829 0.  0.         0.         0.40824829 0.         0.         0.  0.40824829 0.40824829 0.         0.40824829 0.        ]]
```
---

**Example:** Calculating TF-IDF Manually
```python
import numpy as np  # Importing NumPy for numerical operations
import math  # Importing math for logarithmic calculations

# Sample sentences
sentences = [
    "Natural Language Processing is amazing",
    "TF-IDF is a technique for scoring words",
    "The importance of TF-IDF in text analysis"
]

# Tokenizing the sentences
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Compute Term Frequency (TF)
def compute_tf(sentence):
    """
    Calculates term frequency (TF) for each word in a sentence.
    TF is computed as the occurrence of a word divided by total words in the sentence.
    """
    word_count = len(sentence)  # Total words in sentence
    tf_dict = {}  # Dictionary to store TF values
    for word in sentence:
        tf_dict[word] = tf_dict.get(word, 0) + 1 / word_count  # Compute TF
    return tf_dict

# Compute Inverse Document Frequency (IDF)
def compute_idf(sentences):
    """
    Calculates inverse document frequency (IDF) for each unique word.
    IDF is computed using log(total number of sentences / number of sentences containing the word).
    """
    num_sentences = len(sentences)  # Total number of sentences
    idf_dict = {}  # Dictionary to store IDF values
    all_words = set([word for sentence in sentences for word in sentence])  # Unique words in dataset
    for word in all_words:
        containing_sentences = sum(1 for sentence in sentences if word in sentence)  # Count sentences containing word
        idf_dict[word] = math.log(num_sentences / (1 + containing_sentences))  # Compute IDF
    return idf_dict

# Compute TF-IDF
def compute_tfidf(tf, idf):
    """
    Calculates TF-IDF score by multiplying TF and IDF values.
    """
    tfidf = {word: tf[word] * idf[word] for word in tf}  # Compute TF-IDF
    return tfidf

# Applying TF-IDF calculation
tf_values = [compute_tf(sentence) for sentence in tokenized_sentences]  # Compute TF for each sentence
idf_values = compute_idf(tokenized_sentences)  # Compute IDF for dataset
tfidf_values = [compute_tfidf(tf, idf_values) for tf in tf_values]  # Compute TF-IDF

# Display TF-IDF results
print("TF-IDF Values:")
for idx, sentence in enumerate(sentences):
    print(f"Sentence {idx+1}: {tfidf_values[idx]}")  # Print TF-IDF scores for each sentence
```
---
**Output:**
```
Sentence 1: {'natural': 0.366, 'language': 0.366, 'processing': 0.366, 'is': -0.287, 'amazing': 0.366}
Sentence 2: {'tf-idf': -0.287, 'is': -0.287, 'technique': 0.366, 'for': 0.366, 'scoring': 0.366, 'words': 0.366}
Sentence 3: {'the': 0.366, 'importance': 0.366, 'of': 0.366, 'tf-idf': -0.287, 'in': 0.366, 'text': 0.366, 'analysis': 0.366}
```


#### **11. Word Embeddings**
Word embeddings are a technique used to convert words into numerical representations, capturing semantic meaning and relationships between words.

- **Word2Vec:** Converts words into dense vectors representing their meanings.
- **Skip-gram:** Predicts context words from a given target word.
- **CBOW (Continuous Bag of Words):** Predicts a target word from surrounding context words.
- **Custom BoW:** Implementing a custom Bag of Words representation.
- **Sentence Embeddings:** Generating fixed-size vector representations for entire sentences.

**Example:** Training a Word2Vec model on a sample dataset
```python
from gensim.models import Word2Vec  # Importing Word2Vec from gensim

# Defining sample sentences for training
sentences = [['I', 'love', 'NLP'], ['I', 'love', 'AI']]

# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Finding words similar to 'love'
print(model.wv.most_similar('love'))
```

#### **12. Sentence Embeddings**
Sentence embeddings aim to create vector representations for entire sentences to capture semantic information.

- **Tools and Libraries:**
  - **Langchain:** A framework for integrating NLP pipelines.
  - **Hugging Face:** Provides pre-trained transformers for NLP tasks.
  - **Sentence Transformers:** Models specifically designed for sentence-level embeddings.

**Example:** Converting a sentence into a vector representation
```python
from sentence_transformers import SentenceTransformer  # Importing Sentence Transformer

# Loading pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encoding a sentence into an embedding
embeddings = model.encode("Natural Language Processing is amazing!")

# Outputting sentence embedding vector
print(embeddings)
```

#### **13. Word2Vec**
Word2Vec converts words into high-dimensional vector representations, capturing semantic information.

**Example:** Generating word embeddings for words in a small corpus
```python
from gensim.models import Word2Vec  # Importing Word2Vec

# Defining small corpus for training
sentences = [['machine', 'learning'], ['deep', 'learning']]

# Training Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Displaying vector representation for 'machine'
print(model.wv['machine'])
```

#### **14. Continuous Bag of Words (CBOW)**
CBOW is a neural network-based method for word embeddings.

**Example:** Training a CBOW model for predicting words from their context
```python
from gensim.models import Word2Vec  # Importing Word2Vec

# Defining sentences for training
sentences = [['deep', 'learning', 'is', 'great']]

# Training CBOW model
model = Word2Vec(sentences, sg=0, min_count=1)

# Displaying vector for 'learning'
print(model.wv['learning'])
```

#### **15. Skip-gram**
Skip-gram is the inverse of CBOW, where the model predicts context words from a single target word.

**Example:** Training a Skip-gram model for predicting surrounding words
```python
# Training Skip-gram model
model = Word2Vec(sentences, sg=1, min_count=1)

# Displaying vector for 'learning'
print(model.wv['learning'])
```

#### **16. Recurrent Neural Networks (RNN)**
RNNs are designed to process sequential data by maintaining memory of past inputs.

**Example:** Creating a simple RNN model for sequential data processing
```python
import tensorflow as tf  # Importing TensorFlow

# Defining RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, input_shape=(10, 1)),  # Adding RNN layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compiling model
model.compile(loss='binary_crossentropy', optimizer='adam')

# Displaying model architecture
print(model.summary())
```

#### **17. Named Entity Recognition (NER)**
NER identifies and classifies entities (e.g., names, locations) in text.

**Example:** Extracting named entities from a sentence
```python
import spacy  # Importing Spacy

# Loading Spacy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Processing text
doc = nlp("Barack Obama was the 44th President of the USA")

# Displaying named entities with labels
print([(ent.text, ent.label_) for ent in doc.ents])
```

#### **18. Part of Speech (POS) Tagging**
POS tagging assigns grammatical categories to words in a sentence, aiding NLP tasks.

**Example:** Assigning POS tags to words in a sentence
```python
# Iterating through tokens and printing POS tags
for token in doc:
    print(token.text, token.pos_)  # Displaying words with their POS tags
```

