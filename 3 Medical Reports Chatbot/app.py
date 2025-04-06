import os
import streamlit as st
from streamlit_chat import message
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import re

# Apply dark theme with green buttons
st.markdown("""
<style>
body, .stApp {
    background-color: #0E0E0E;
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
}

/* Input Fields */
input, textarea {
    background-color: #1E1E1E !important;
    color: #E0E0E0 !important;
    border: 1px solid #333333 !important;
}

label {
    color: #E0E0E0 !important;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Forest Green Buttons - Target all buttons more aggressively */
button[kind="primary"], .stButton > button {
    background-color: #228B22 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 0.6em 1.4em !important;
    border: none !important;
    transition: background-color 0.3s ease;
}

button[kind="primary"]:hover, .stButton > button:hover {
    background-color: #1B6E1B !important;
    color: white !important;
}

button[kind="primary"]:disabled, .stButton > button:disabled {
    background-color: #3a3a3a !important;
    color: #aaa !important;
    border-radius: 8px !important;
    opacity: 0.6;
}

.user-container {
    text-align: right;
    margin-right: 10px;
}
.bot-container {
    text-align: left;
    margin-left: 10px;
}
.user-bubble, .bot-bubble {
    padding: 10px;
    margin: 5px 0;
    border-radius: 10px;
    font-size: 16px;
    max-width: 80%;
    display: inline-block;
}
.user-bubble {
    background-color: #03DAC6;
    color: #121212;
}
.bot-bubble {
    background-color: #1E1E1E;
    color: #E0E0E0;
}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize GROQ LLM
chat_gpt = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    groq_api_key=groq_api_key
)

# SQLite database setup
DATABASE_URL = "sqlite:///my_database.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)

users_table = Table("users", metadata, autoload_with=engine)
reports_table = Table("medical_reports", metadata, autoload_with=engine)
SessionLocal = sessionmaker(bind=engine)

def check_user(username, password):
    session = SessionLocal()
    user = session.query(users_table).filter_by(username=username, password=password).first()
    session.close()
    return user

def fetch_reports(username):
    session = SessionLocal()
    user_reports = session.query(reports_table).filter_by(username=username).all()
    session.close()
    return user_reports

# Function to detect if the input is Japanese
def is_japanese(text):
    # Check for hiragana, katakana, or kanji characters
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

def get_response(history, user_message, medical_records):
    # Detect input language
    is_input_japanese = is_japanese(user_message)

    # Define few-shot examples to guide the model
    few_shot_examples = """
    Example 1:
    Human: Can you tell me about my recent blood test results?
    Medical Expert: Your recent blood test results show normal levels of hemoglobin and glucose.

    Example 2:
    Human: 最近の血液検査の結果を教えてください。
    Medical Expert: 最近の血液検査の結果では、ヘモグロビンとグルコースの値が正常範囲内にあります。
    """

    # Construct the prompt dynamically based on language
    if is_input_japanese:
        DEFAULT_TEMPLATE = """
        You are a medical expert with access to all medical records. The user can ask about past procedures, doctor reports, test results, and symptom clarifications. Respond in a helpful and friendly manner. The user's input is in Japanese, so you must respond in Japanese.

        Here are some examples to guide your response:
        {few_shot_examples}

        Conversation history:
        {context}

        User's Medical Records:
        {text}

        Current conversation:
        Human: {input}
        Medical Expert:
        """
    else:
        DEFAULT_TEMPLATE = """
        You are a medical expert with access to all medical records. The user can ask about past procedures, doctor reports, test results, and symptom clarifications. Respond in a helpful and friendly manner. The user's input is in English, so you must respond in English.

        Here are some examples to guide your response:
        {few_shot_examples}

        Conversation history:
        {context}

        User's Medical Records:
        {text}

        Current conversation:
        Human: {input}
        Medical Expert:
        """

    PROMPT = PromptTemplate(
        input_variables=["context", "input", "text", "few_shot_examples"],
        template=DEFAULT_TEMPLATE
    )

    chain = LLMChain(llm=chat_gpt, prompt=PROMPT)
    return chain.predict(context=history, input=user_message, text=medical_records, few_shot_examples=few_shot_examples)

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "username" not in st.session_state:
    st.session_state.username = ""

def login_page():
    st.title("🔐 医療レポートログイン")
    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type="password")

    if st.button("ログイン"):
        user = check_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("ログインに成功しました！")
            st.rerun()
        else:
            st.error("ユーザー名またはパスワードが無効です")

def chatbot_page():
    st.title("💬 医療専門家とチャット")
    reports = fetch_reports(st.session_state.username)
    medical_records = "\n".join([
        f"日付: {report.date}, 詳細: {report.report}" for report in reports
    ]) if reports else "記録が見つかりません。"

    for i, chat in enumerate(st.session_state.conversation_history):
        is_user = chat["sender"] == "User"
        emoji = "" if is_user else "🤖"  # Remove emoji for user, keep for bot
        container_class = "user-container" if is_user else "bot-container"
        bubble_class = "user-bubble" if is_user else "bot-bubble"
        st.markdown(f"""
            <div class="{container_class}">
                <div class="{bubble_class}">{emoji} {chat['message']}</div>
            </div>
        """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("医療に関する質問をしてください...", key="user_input")
        submitted = st.form_submit_button("🚀")

    if submitted and user_input.strip():
        st.session_state.conversation_history.append({"sender": "User", "message": user_input})
        bot_reply = get_response(
            history=st.session_state.conversation_history,
            user_message=user_input,
            medical_records=medical_records
        )
        st.session_state.conversation_history.append({"sender": "Chatbot", "message": bot_reply})
        st.rerun()

def main():
    if st.session_state.logged_in:
        chatbot_page()
    else:
        login_page()

if __name__ == "__main__":
    main()