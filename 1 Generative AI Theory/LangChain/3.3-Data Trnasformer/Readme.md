# Contributions to Logic and Set Theory

This briefing document synthesizes information from three sources, focusing on different text splitting methodologies and the significant mathematical and philosophical contributions of Kurt Gödel.

## Part 1: Text Splitting Methodologies

This section reviews various approaches to splitting text, highlighting their mechanisms and how chunk size is determined.

### 1. HTML Header Text Splitter

The HTMLHeaderTextSplitter is a "structure-aware" chunking tool designed to split text at the HTML element level. Its primary goal is to maintain the semantic grouping of related text and preserve context-rich information embedded within document structures.

- **How Text is Split**: By HTML element (e.g., h1, h2, h3).
- **Chunking Objective**: To keep related text grouped semantically and retain structural context.
- **Metadata**: It adds metadata for each header relevant to a given chunk.
- **Flexibility**: It can return chunks element by element or combine elements with the same metadata.
- **Pipeline Integration**: Can be used in conjunction with other text splitters.
- **Example from Source**: A demonstration shows how an HTML string with h1, h2, and h3 tags is split, and each resulting document includes metadata indicating its header hierarchy. For instance, a chunk of text under "Bar subsection 1" would have metadata `{'Header 1': 'Foo', 'Header 2': 'Bar main section', 'Header 3': 'Bar subsection 1'}`.

### 2. Character Text Splitter

The CharacterTextSplitter is described as the simplest text splitting method.

- **How Text is Split**: By a single character separator, defaulting to "\n\n".
- **How Chunk Size is Measured**: By the number of characters.
- **Behavioral Note**: The example demonstrates that if chunk_size is specified, the splitter might still create chunks larger than the specified size if the separator "\n\n" results in larger blocks of text.

### 3. Recursive Character Text Splitter

This is the recommended text splitter for generic text due to its hierarchical splitting approach.

- **How Text is Split**: It is parameterized by a list of characters (defaulting to ["\n\n", "\n", " ", ""]). It attempts to split text using these separators in order, from the most significant (like double newlines for paragraphs) to the least significant (single spaces or no separator at all), until chunks are small enough.
- **How Chunk Size is Measured**: By the number of characters.
- **Semantic Preservation**: This method aims to keep paragraphs, sentences, and words together as long as possible, assuming these are the "strongest semantically related pieces of text."

### 4. Recursive JSON Splitter

This splitter is specifically designed for JSON data, allowing control over chunk sizes while attempting to maintain the integrity of nested JSON objects.

- **How Text is Split**: By JSON value, traversing the JSON data depth-first to build smaller JSON chunks.
- **Chunking Objective**: To keep nested JSON objects whole, but it will split them if necessary to adhere to min_chunk_size and max_chunk_size limits.
- **Handling Large Strings**: If a value is a very large string and not a nested JSON object, the string itself will not be split by this splitter. It is suggested to compose this splitter with a Recursive Text splitter for such cases if a hard cap on chunk size is needed.
- **List Pre-processing**: There is an optional pre-processing step to split lists by converting them to JSON (dict) format and then splitting them.
- **How Chunk Size is Measured**: By the number of characters.
- **Output**: Can also output documents.
