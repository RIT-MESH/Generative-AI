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

#### **10. Diagrams and Workflows**
The source includes illustrations of:
- **RAG architecture**
- **Self-ask prompting techniques**
- **Discovery process flow**

