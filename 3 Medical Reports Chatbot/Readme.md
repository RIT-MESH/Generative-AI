# Medical Reports Chatbot Documentation

## Overview

The **Medical Reports Chatbot** is a web-based application built using Streamlit, designed to allow users to access their medical records and interact with a medical expert chatbot. The chatbot provides information about past procedures, doctor reports, test results, and symptom clarifications based on the user's medical records. The application supports bilingual interaction: users can communicate in English or Japanese, and the chatbot responds in the same language as the input.

### Key Features
- **User Authentication**: Users must log in with a username and password to access their medical records.
- **Medical Records Access**: The application fetches medical records from a SQLite database.
- **Bilingual Chatbot**: The chatbot responds in the same language as the user's input (English or Japanese) using prompt engineering and few-shot examples.
- **Custom Styling**: The application uses a dark theme with green buttons for a modern look.
- **Chat Interface**: A user-friendly chat interface displays conversation history with distinct styling for user and bot messages.

## Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.10+**
- **Pip** (Python package manager)
- Required Python packages (install via `requirements.txt`):
  ```
  streamlit==1.30.0
  streamlit-chat==0.1.1
  langchain==0.3.1
  langchain-community==0.3.1
  langchain-core==0.3.6
  langchain-groq==0.2.0
  langchain-huggingface==0.1.0
  openai==1.50.2
  tiktoken==0.7.0
  emoji==2.14.0
  python-dotenv==1.0.1
  sqlalchemy==2.0.35
  groq==0.11.0
  ```

### Installation Steps

1. **Clone the Repository** 
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Create a `requirements.txt` file with the above packages and run:
   ```
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the project root directory.
   - Add your Groq API key:
     ```
     GROQ_API_KEY=<your-groq-api-key>
     ```
   - You can obtain a Groq API key from [Groq's website](https://groq.com).

4. **Set Up the Database**:
   - The application uses a SQLite database (`my_database.db`) to store user credentials and medical records.
   - Ensure the database file exists in the project root directory. If it doesn't, you need to create it with the following schema:
     - **users** table: Stores user credentials.
       ```sql
       CREATE TABLE users (
           username TEXT PRIMARY KEY,
           password TEXT NOT NULL
       );
       ```
     - **medical_reports** table: Stores medical records.
       ```sql
       CREATE TABLE medical_reports (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           username TEXT,
           date TEXT,
           report TEXT,
           FOREIGN KEY (username) REFERENCES users(username)
       );
       ```
   - You can use a tool like [DB Browser for SQLite](https://sqlitebrowser.org/) to create and populate the database with sample data.

## Project Structure

The project consists of a single Python script (`app.py`) with the following structure:

- **Imports and Setup**:
  - Imports necessary libraries (Streamlit, SQLAlchemy, LangChain, Groq, etc.).
  - Applies custom CSS for styling using `st.markdown`.
  - Loads environment variables using `python-dotenv` and initializes the Groq LLM.

- **Database Setup**:
  - Connects to a SQLite database using `sqlalchemy` to manage user credentials and medical records.

- **Core Functions**:
  - `check_user`: Validates user login credentials.
  - `fetch_reports`: Retrieves medical records for a given user.
  - `is_japanese`: Detects if the input text is in Japanese.
  - `get_response`: Generates a chatbot response based on user input, conversation history, and medical records.

- **Pages**:
  - `login_page`: Displays the login interface.
  - `chatbot_page`: Displays the chat interface for interacting with the medical expert.

- **Main Function**:
  - Orchestrates the application flow based on the user's login state.

## Code Details

### 1. Dependencies and Their Roles
- **streamlit**: Core framework for building the web application.
- **streamlit-chat**: Provides a chat interface component for displaying conversation history.
- **langchain**: Framework for building applications with LLMs, used for prompt engineering and chaining.
- **langchain-community**: Community extensions for LangChain (not directly used but required as a dependency).
- **langchain-core**: Core components of LangChain (required for `langchain` and `langchain-groq`).
- **langchain-groq**: LangChain integration for Groq’s LLM API, used to interact with the `llama3-8b-8192` model.
- **langchain-huggingface**: LangChain integration for Hugging Face models (not used in this project but included in dependencies).
- **openai**: OpenAI API client (not used in this project but included in dependencies).
- **tiktoken**: Tokenization library for OpenAI models (not used but included in dependencies).
- **emoji**: Library for handling emojis in the chat interface (used for the bot emoji `🤖`).
- **python-dotenv**: Loads environment variables from a `.env` file.
- **sqlalchemy**: ORM for interacting with the SQLite database.
- **groq**: Official Groq API client (used indirectly via `langchain-groq`).

### 2. Styling
The application uses a dark theme with green buttons (`#228B22`). The CSS is applied via `st.markdown` and includes:
- Dark background (`#0E0E0E`) and light text (`#E0E0E0`) for readability.
- Custom styling for input fields, labels, and buttons.
- Green buttons with a hover effect (changes to `#1B6E1B`).
- Distinct chat bubble styles:
  - User messages: Cyan background (`#03DAC6`) with dark text (`#121212`).
  - Bot messages: Dark gray background (`#1E1E1E`) with light text (`#E0E0E0`).

### 3. Database Integration
- **Database**: SQLite (`my_database.db`).
- **Tables**:
  - `users`: Stores usernames and passwords.
  - `medical_reports`: Stores medical records with fields for `id`, `username`, `date`, and `report`.
- **Functions**:
  - `check_user`: Queries the `users` table to validate login credentials.
  - `fetch_reports`: Queries the `medical_reports` table to fetch records for the logged-in user.

### 4. Chatbot Logic
- **Language Detection**:
  - The `is_japanese` function uses a regular expression to detect Japanese characters (hiragana, katakana, kanji).
  - Example: `bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))`

- **Prompt Engineering**:
  - The `get_response` function dynamically constructs a prompt based on the input language.
  - It uses few-shot examples to guide the model in responding in the correct language:
    - English input → English response.
    - Japanese input → Japanese response.
  - The prompt explicitly instructs the model to respond in the same language as the input:
    - For Japanese: "The user's input is in Japanese, so you must respond in Japanese."
    - For English: "The user's input is in English, so you must respond in English."

- **LLM**:
  - Uses the `llama3-8b-8192` model from Groq via the `ChatGroq` class.
  - Configured with `temperature=0` for deterministic responses.

### 5. Session Management
- Uses Streamlit's `session_state` to manage:
  - `logged_in`: Tracks whether the user is logged in (`True`/`False`).
  - `username`: Stores the logged-in user's username.
  - `conversation_history`: Stores the chat history as a list of dictionaries with `sender` ("User" or "Chatbot") and `message`.

### 6. Pages
- **Login Page** (`login_page`):
  - Displays a login form with username and password fields.
  - Validates credentials using `check_user`.
  - Displays success or error messages using `st.success` or `st.error`.
  - Redirects to the chat page upon successful login using `st.rerun`.

- **Chatbot Page** (`chatbot_page`):
  - Displays the chat interface with conversation history.
  - Fetches medical records using `fetch_reports`.
  - Allows users to input questions and receive responses from the chatbot.
  - Uses a form (`st.form`) for user input to prevent page reloads on submission.
  - Updates the conversation history and refreshes the page using `st.rerun`.

## Usage Instructions

1. **Run the Application**:
   - Navigate to the project directory and run:
     ```
     streamlit run app.py
     ```
   - This will start a local server, and a browser window will open with the application (typically at `http://localhost:8501`).

2. **Log In**:
   - On the login page, enter your username and password.
   - If the credentials are correct, you’ll be redirected to the chat page. If incorrect, an error message will be displayed.

3. **Interact with the Chatbot**:
   - On the chat page, type your question in the input box (e.g., "Can you tell me about my recent blood test results?" or "最近の血液検査の結果を教えてください").
   - The chatbot will respond in the same language as your input:
     - English input → English response.
     - Japanese input → Japanese response.
   - The conversation history will be displayed, with user messages in cyan and bot messages in dark gray with a `🤖` emoji.

4. **View Medical Records**:
   - The chatbot has access to your medical records and can provide details based on your questions.
   - Records are fetched from the `medical_reports` table and formatted as "日付: {date}, 詳細: {report}".

## Limitations

- **Model Language Capability**:
  - The `llama3-8b-8192` model is primarily trained on English data, so Japanese responses may be less fluent or natural compared to English responses.
  - The model may occasionally respond in the wrong language, especially for English inputs, due to the influence of Japanese prompts or context.

- **Unused Dependencies**:
  - Some dependencies (`langchain-huggingface`, `openai`, `tiktoken`) are not used in the current implementation but are included in the dependency list. These can be removed if not needed.

- **Database**:
  - The application assumes the SQLite database is pre-populated with user credentials and medical records. There is no interface to add or edit records within the app.

- **Performance**:
  - The application may experience delays when fetching responses from the Groq API, depending on network conditions and API response times.

## Future Improvements

1. **Multilingual Model**:
   - Replace the current model with a more robust multilingual model (e.g., XLM-RoBERTa) to improve Japanese response quality and language consistency.

2. **Database Management**:
   - Add an admin interface to manage users and medical records (e.g., add, edit, delete records).

3. **Language Detection**:
   - Improve language detection by using a more sophisticated library (e.g., `langdetect`) to handle edge cases better.

4. **Error Handling**:
   - Add more robust error handling for database connections, API requests, and user inputs.

5. **UI Enhancements**:
   - Add a logout button to allow users to return to the login page.
   - Improve the chat interface with features like message timestamps or a "clear chat" button.

6. **Optimize Dependencies**:
   - Remove unused dependencies (`langchain-huggingface`, `openai`, `tiktoken`) to reduce the project’s footprint.

## Troubleshooting

- **Login Fails**:
  - Ensure the `users` table in the database contains the correct username and password.
  - Check for typos in the `.env` file for the `GROQ_API_KEY`.

- **Chatbot Responds in Wrong Language**:
  - The model may struggle with language consistency due to its training data. Consider switching to a multilingual model or adding more few-shot examples to the prompt.

- **Application Doesn’t Load**:
  - Verify that all dependencies are installed (`pip install -r requirements.txt`).
  - Ensure the SQLite database file (`my_database.db`) exists in the project root directory.

- **Chatbot Doesn’t Respond**:
  - Check your internet connection, as the Groq API requires an active connection.
  - Verify that the `GROQ_API_KEY` is valid and has not expired.

## License

This project is for educational and personal use. Ensure compliance with the terms of service for Groq API and any other third-party libraries used.

---

