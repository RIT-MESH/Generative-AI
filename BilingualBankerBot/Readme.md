# BilingualBankerBot

This project details the creation of a bilingual (English/Japanese) banking chatbot using AWS Lex V2 for conversation management, AWS Lambda for business logic, and Amazon Bedrock (with Anthropic Claude Instant) for enhancing some responses with generative AI. The bot simulates balance checks and internal fund transfers and is configured for the `ap-northeast-1` (Tokyo) AWS region.

**Features:**


https://github.com/user-attachments/assets/0b360746-e929-4ac0-91dd-ea6e6bed68fe


https://github.com/user-attachments/assets/14099d19-9209-4645-80bd-33e27c6d4a93





---

**Features:**

*   **Bilingual Support:** English (en-US) and Japanese (ja-JP).
*   **Core Intents:**
    *   `WelcomeIntent`: Greets the user.
    *   `CheckBalance`: Simulates checking account balance (uses direct template for JA, Bedrock for EN).
    *   `FollowupCheckBalance`: Handles follow-up balance checks using context (direct template for JA, Bedrock for EN).
    *   `MakeTransfer`: Simulates internal fund transfers with a custom confirmation prompt managed by Lambda (direct template for final success/cancel messages).
    *   `FallbackIntent`: Handles unrecognized input using Bedrock.
*   **Generative AI Enhancement:** Uses Amazon Bedrock (Claude Instant) for more natural balance responses (English only) and fallback messages.
*   **Lambda Validation:** Implements date format validation for `CheckBalance` & `FollowupCheckBalance` using a Lambda code hook.
*   **Custom Confirmation:** `MakeTransfer` intent uses a Lambda-driven custom confirmation prompt to ensure full Japanese localization.
*   **Region Specific:** Designed for `ap-northeast-1` (Tokyo).

---

## Phase 0: Preparation & AWS Account Setup

1.  **AWS Account:** Ensure you have an active AWS account.
2.  **Region Selection:** Log in to the AWS Management Console. In the top-right corner, select the **Asia Pacific (Tokyo) `ap-northeast-1`** region. ALL subsequent steps must be performed in this region.
3.  **Text Editor:** Have a text editor (like VS Code, Sublime Text, etc.) ready for copying code.
4.  **Bedrock Model Access:**
    *   Navigate to the **Amazon Bedrock** service in the AWS Console (in `ap-northeast-1`).
    *   In the left navigation, click **Model access**.
    *   Click **Manage model access**.
    *   Request access to **Anthropic -> Claude Instant 1.2** (Model ID: `anthropic.claude-instant-v1`).
    *   **WAIT** until the status shows **"Access granted"**.
    *   Note the **Model ID:** `anthropic.claude-instant-v1`.

---

## Phase 1: Foundational AWS Setup (IAM Roles)

Create the necessary IAM roles to allow AWS services to interact securely.

**Step 1.1: Create IAM Role for Lex Bot**

1.  Navigate to **IAM** in the AWS Console -> **Roles** -> **Create role**.
2.  **Trusted entity type:** Select `AWS service`.
3.  **Use case:** Find and select `Lex`. Choose `Lex V2 Bot` as the specific use case. Click **Next**.
4.  **Add permissions:** Click **Next** (we'll add specific permissions later).
5.  **Name, review, and create:**
    *   **Role name:** `LexV2BotRole-ProBilingual-Tokyo`
    *   **Description:** `Custom Role for ProBilingualBankerBot-Tokyo Lex bot.`
    *   Click **Create role**.

**Step 1.2: Create IAM Role for Lambda Function**

1.  Go back to **IAM -> Roles** -> **Create role**.
2.  **Trusted entity type:** `AWS service`.
3.  **Use case:** Find and select `Lambda`. Click **Next**.
4.  **Add permissions:**
    *   Search for and check the box next to `AWSLambdaBasicExecutionRole` (allows writing logs to CloudWatch).
    *   Click **Next**.
5.  **Name, review, and create:**
    *   **Role name:** `LambdaExecutionRole-ProBilingual-Tokyo`
    *   **Description:** `Execution role for ProBilingualBankerHandler-Tokyo Lambda.`
    *   Click **Create role**.

**Step 1.3: Add Bedrock Permission to Lambda Role**

1.  Go back to **IAM -> Roles**.
2.  Find and click on `LambdaExecutionRole-ProBilingual-Tokyo`.
3.  On the **Permissions** tab, click **Add permissions** -> **Create inline policy**.
4.  Select the **JSON** tab. Delete existing content.
5.  Paste the following policy:
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": "arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-instant-v1"
            }
        ]
    }
    ```
6.  Click **Review policy**.
7.  **Name:** `BedrockInvokeClaudeInstantTokyoPolicy`.
8.  Click **Create policy**.

---

## Phase 2: Build the Bilingual Lex V2 Bot

**Step 2.1: Create Bot Shell**

1.  Navigate to **Amazon Lex V2** console (in `ap-northeast-1`).
2.  Click **Create bot** -> **Create a blank bot**.
3.  **Bot name:** `ProBilingualBankerBot-Tokyo`
4.  **Description:** `EN/JA Banker Bot + Bedrock (Tokyo)`
5.  **IAM permissions:** Select `Use an existing role` -> Choose `LexV2BotRole-ProBilingual-Tokyo`.
6.  **COPPA:** `No`. **Idle session timeout:** `5 minutes`. Click **Next**.
7.  **Add Language 1:** `English (US)`. Voice: e.g., `Matthew`.
8.  **Add Language 2:** Click `Add language`. Select `Japanese (Japan)`. Voice: e.g., `Mizuki`.
9.  **Confidence score threshold:** `0.40`. Click **Done**.

**Step 2.2: Create `WelcomeIntent`**

*   Configure for both **English (US)** and **Japanese (Japan)** tabs.
*   **Intent name:** `WelcomeIntent`. **Description:** `Greets user`.
*   **EN Utterances:** `Hi`, `Hello`, `Hey`. **EN Closing Msg:** `Hello! I'm your AI Banker Bot. How can I assist?`
*   **JA Utterances:** `こんにちは`, `もしもし`. **JA Closing Msg:** `こんにちは！AIバンカーボットです。ご用件は何でしょうか？`
*   Save for both languages.

**Step 2.3: Customize `FallbackIntent`**

*   Configure for both **English (US)** and **Japanese (Japan)** tabs.
*   **EN Closing Msg:** `Sorry, I can help with balance checks and transfers. Please rephrase.`
*   **JA Closing Msg:** `すみません、残高照会と振替についてお手伝いできます。言い換えていただけますか？`
*   Save for both languages.

**Step 2.4: Create `AccountType` Custom Slot Type**

1.  Left menu -> **Slot types** -> **Add slot type** -> **Add blank slot type**.
2.  **Name:** `AccountType`. **Desc:** `Bank account types`. **Resolution:** `Restrict to slot values`.
3.  **Values:**
    *   Value: `Checking`. EN Syn: `check`, `current`. JA Syn: `普通預金`, `普通`.
    *   Value: `Savings`. EN Syn: `save`. JA Syn: `貯蓄預金`, `貯蓄`.
    *   Value: `Credit`. EN Syn: `credit card`, `visa`. JA Syn: `クレジットカード`, `クレカ`.
4.  Click **Save slot type**.

**Step 2.5: Create `CheckBalance` Intent**

*   Left menu -> **Intents** -> **Add intent** -> **Add empty intent**.
*   **Name:** `CheckBalance`. **Desc:** `Checks balance`.
*   **Configure EN (US):**
    *   Utterances: `check balance`, `what is my balance`, `check my {accountType} balance`, `how much in {accountType}`.
    *   Slots:
        *   1: Name `accountType`, Type `AccountType`, Prompt `Checking, Savings, or Credit account?`. **Required.**
        *   2: Name `dateOfBirth`, Type `AMAZON.Date`, Prompt `For verification, what is your date of birth? (YYYY-MM-DD)`. **Required.**
    *   Contexts (Output): Name `contextCheckBalance`, Timeout `5` turns / `90` seconds. Add.
    *   Save.
*   **Configure JA (JP):**
    *   Utterances: `残高照会`, `{accountType} の残高`.
    *   Slots: Edit Prompts: `accountType` -> `どちらの口座ですか？（普通預金、貯蓄預金、クレジットカード）`, `dateOfBirth` -> `確認のため、生年月日を教えてください。（YYYY-MM-DD）`.
    *   Save.

**Step 2.6: Create `FollowupCheckBalance` Intent**

*   Left menu -> **Intents** -> **Add intent** -> **Add empty intent**.
*   **Name:** `FollowupCheckBalance`. **Desc:** `Follow-up balance check`.
*   **Contexts (Input):** Choose contexts -> Select `contextCheckBalance`.
*   **Configure EN (US):**
    *   Utterances: `how about my {accountType} account?`, `what about {accountType}?`.
    *   Slots:
        *   1: Name `accountType`, Type `AccountType`, Prompt `Which other account?`. **Required.**
        *   2: Name `dateOfBirth`, Type `AMAZON.Date`, Prompt `Just to re-confirm, your date of birth?`. **Required.**
            *   Edit `dateOfBirth` slot -> Advanced options -> Default values -> Enter `#contextCheckBalance.dateOfBirth` -> Add default value -> Update slot.
    *   Save.
*   **Configure JA (JP):**
    *   Utterances: `{accountType} 口座についてはどうですか？`, `{accountType} は？`.
    *   Slots: Edit Prompts: `accountType` -> `他のどの口座ですか？`, `dateOfBirth` -> `念のため再度、生年月日をお願いします。`.
    *   Save.

**Step 2.7: Create `MakeTransfer` Intent**

*   Left menu -> **Intents** -> **Add intent** -> **Add empty intent**.
*   **Name:** `MakeTransfer`. **Desc:** `Transfers funds`.
*   **Configure Slots (under EN tab first):**
    *   1: Name `amount`, Type `AMAZON.Number`, Prompt (EN) `How much?`. **Required.**
    *   2: Name `sourceAccount`, Type `AccountType`, Prompt (EN) `Transfer FROM which account?`. **Required.**
    *   3: Name `destinationAccount`, Type `AccountType`, Prompt (EN) `Transfer TO which account?`. **Required.**
*   **Configure EN (US):**
    *   Utterances: `transfer money`, `transfer {amount} from {sourceAccount} to {destinationAccount}`.
    *   **Confirmation:** **TURN OFF** the "Active" toggle.
    *   Save.
*   **Configure JA (JP):**
    *   Utterances: `振替`, `{sourceAccount} から {destinationAccount} へ {amount} 振り替え`.
    *   Slots: Edit Prompts: `amount` -> `金額は？`, `sourceAccount` -> `どの口座から？`, `destinationAccount` -> `どの口座へ？`.
    *   Save.

**Step 2.8: Initial Bot Build**

1.  Click **Build**. Wait 1-2 minutes.

---

## Phase 3: Backend Logic - Lambda Function (Bilingual + Bedrock)

**Step 3.1: Create Lambda Function**

1.  Navigate to **AWS Lambda** (in `ap-northeast-1`).
2.  Click **Create function** -> **Author from scratch**.
3.  Function name: `ProBilingualBankerHandler-Tokyo`.
4.  Runtime: `Python 3.11` (or 3.12).
5.  Permissions: `Use an existing role` -> Select `LambdaExecutionRole-ProBilingual-Tokyo`.
6.  Advanced settings: Edit -> Timeout: `15` seconds -> Save.
7.  Click **Create function**.

**Step 3.2: Add Lambda Code**

1.  Go to the **Code** tab of `ProBilingualBankerHandler-Tokyo`.
2.  Delete all code in `lambda_function.py`.
3.  **COPY** the **ENTIRE** Python code block below and **PASTE** it.
4.  **VERIFY** `BEDROCK_MODEL_ID = "anthropic.claude-instant-v1"` and `AWS_DEFAULT_REGION = "ap-northeast-1"`.
5.  Click **Deploy**.

``` 
# --- Start Complete FINAL Lambda Code (Syntax Corrected for invoke_bedrock & All Helpers) ---
import json 
import random
import decimal
import boto3
import logging
import os
import datetime
import re # Import re for get_localized_text helper

# --- Logging Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Use INFO for production, DEBUG for development tracing

# --- Configuration ---
BEDROCK_MODEL_ID = "anthropic.claude-instant-v1"
AWS_DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")

# Initialize AWS Clients
try:
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_DEFAULT_REGION)
    logger.info(f"Initialized Bedrock client for region {AWS_DEFAULT_REGION}")
except Exception as init_error:
    logger.error(f"FATAL: Could not initialize boto3 Bedrock client: {init_error}", exc_info=True)
    bedrock_runtime = None

# --- Bilingual Prompts & Messages ---
PROMPTS_AND_MESSAGES = {
    'en-US': {
        'balance_prompt': "You are a polite bank teller AI assistant responding to a balance inquiry. The user's {account_type} account balance is ${balance}. State this clearly using digits and symbols (like $1,234.56) in a friendly tone.",
        'fallback_prompt': "You are a helpful banking assistant primarily focused on balance checks and fund transfers for users in Japan. The user said: '{user_input}'. Briefly try to answer if it's a very general banking question using only common knowledge (do not invent specifics like interest rates). If you cannot answer, or it's not a general banking question, politely state you can primarily assist with balance checks and fund transfers and ask the user to rephrase.",
        'transfer_confirmed_simple': "Okay, the simulated transfer of ${amount} from {source_account} to {destination_account} is complete. Your confirmation number is {confirmation_number}.",
        'transfer_cancelled': "Okay, the transfer request has been cancelled.",
        'transfer_confirmation_custom': "Okay, please confirm: Transfer ${amount} from {source_account} to {destination_account}? (Yes/No)",
        'default_error': "I'm sorry, there was an issue processing your request at this moment. Please try again.",
        'bedrock_error': "I'm having a little trouble generating a detailed response right now, but I can confirm the basic details. Alternatively, please try again shortly.",
        'ask_dob': "For verification, please provide your date of birth in YYYY-MM-DD format.",
        'balance_confirmed_simple': "Okay, the current balance for your {account_type} account is ${balance}."
    },
    'ja_JP': { # Ensure key is ja_JP (underscore)
        'balance_prompt': "あなたは丁寧な銀行のAIアシスタントです。残高照会の応答をしています。ユーザーの {account_type} 口座の残高は ¥{balance} です。これを明確かつ簡潔に、算用数字と記号（例：¥123,456）を使用して、親しみやすいトーンで述べてください。",
        'fallback_prompt': "あなたは日本のユーザー向けの親切な銀行アシスタントで、主に残高照会と資金振替を扱います。ユーザーは次のように言いました：「{user_input}」。もし一般的な銀行業務に関する非常に簡単な質問であれば、一般的な知識のみを使用して簡潔に答えてみてください（金利などの具体的な情報は捏造しないでください）。答えられない場合、または一般的な銀行の質問でない場合は、主に残高照会と資金振替のサポートが可能であることを丁寧に述べ、ユーザーに言い換えをお願いしてください。",
        'transfer_confirmed_simple': "承知いたしました。{source_account} 口座から {destination_account} 口座への ¥{amount} の模擬振込が完了しました。確認番号は {confirmation_number} です。",
        'transfer_cancelled': "承知しました、振替リクエストはキャンセルされました。",
        'transfer_confirmation_custom': "確認：{source_account} 口座から {destination_account} 口座へ ¥{amount} 振り替えます。よろしいですか？ (はい/いいえ)",
        'default_error': "申し訳ありません、リクエストの処理中に問題が発生しました。もう一度お試しください。",
        'bedrock_error': "申し訳ありません、現在詳細な応答を生成するのに少し問題が発生していますが、基本的な情報は確認できます。または、しばらくしてからもう一度お試しください。",
        'ask_dob': "確認のため、生年月日をYYYY-MM-DD形式で教えてください。",
        'balance_confirmed_simple': "承知いたしました。{account_type}口座の現在の残高は ¥{balance} です。"
    }
}
JA_ACCOUNT_MAP = {"Checking": "普通預金", "Savings": "貯蓄預金", "Credit": "クレジット"}

# --- Helper Functions (Correct Syntax/Indentation) ---

def is_valid_iso_date(date_string):
    """Checks if a string strictly matches YYYY-MM-DD format and is valid."""
    if not date_string or not isinstance(date_string, str):
        return False
    if len(date_string) != 10 or date_string.count('-') != 2:
        return False
    try:
        year, month, day = map(int, date_string.split('-'))
        datetime.datetime(year, month, day) # Validate date components
        # Validate year range
        if 1900 < year < datetime.datetime.now().year + 1:
             return True
        else:
             logger.warning(f"Year out of range: {year}")
             return False
    except ValueError:
        return False # Handle parse errors or invalid dates

def get_localized_text(locale, key, **kwargs):
    """Gets localized text template and formats basic placeholders."""
    texts_dict=PROMPTS_AND_MESSAGES
    final_locale = locale if locale in texts_dict else 'en-US'
    lang_texts = texts_dict[final_locale]
    template = lang_texts.get(key, lang_texts.get('default_error', texts_dict['en-US']['default_error']))
    try:
        # Use regex to find placeholders like {key_name}
        placeholders = re.findall(r'\{(\w+)\}', template)
        # Build args safely, providing fallback for missing kwargs
        format_args = {}
        for p in placeholders:
             format_args[p]=kwargs.get(p,f"[{p.upper()}_MISSING]") # Use get() for safety
        return template.format(**format_args)
    except Exception as e:
        logger.error(f"Formatting error locale '{locale}', key '{key}', kwargs '{kwargs}': {e}", exc_info=True)
        # Return a safe default error message
        error_lang_texts = texts_dict.get(locale, texts_dict['en-US'])
        return error_lang_texts.get('default_error', "An error occurred.")

def invoke_bedrock(prompt):
    """Invokes Bedrock model (Claude Instant). Returns text or None."""
    if not bedrock_runtime:
        logger.error("Bedrock runtime client not initialized. Cannot invoke model.")
        return None

    logger.info(f"Invoking Bedrock model {BEDROCK_MODEL_ID} in region {AWS_DEFAULT_REGION}")
    # Format prompt correctly for Claude Instant
    formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
    # Be cautious logging potentially sensitive prompt data in production
    logger.debug(f"Formatted Prompt first 100 chars: {formatted_prompt[:100]}...")

    try:
        # Body structure specifically for Claude Instant/V2
        body = json.dumps({
            "prompt": formatted_prompt,
            "max_tokens_to_sample": 300,
            "temperature": 0.5,
            "stop_sequences": ["\n\nHuman:"]
        })

        # Make the API call
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=BEDROCK_MODEL_ID,
            accept='application/json',
            contentType='application/json'
            )

        # Process the response
        response_body = json.loads(response['body'].read())
        logger.info(f"Bedrock Raw Response Body: {json.dumps(response_body)}") # Log for debugging

        # Extract text ('completion' key for Claude Instant/V2)
        generated_text = response_body.get('completion')

        if not generated_text:
            logger.warning("Bedrock response parsing failed ('completion' key missing or empty).")
            return None # Return None if no text found

        # Clean and return the text
        cleaned_text = generated_text.strip()
        logger.info(f"Bedrock Generated Text (Cleaned): {cleaned_text}")
        return cleaned_text

    except Exception as e:
        # Log any errors during the invocation or processing
        logger.error(f"Bedrock invocation error for model {BEDROCK_MODEL_ID}: {e}", exc_info=True)
        return None # Return None on any exception

# --- Lex Response Helpers ---
def close(session_attributes, intent_name, fulfillment_state, message, locale):
    """Builds a Close response for Lex V2."""
    logger.info(f"Closing intent: {intent_name}, State: {fulfillment_state}")
    response_message = message if message else get_localized_text(locale, 'default_error')
    logger.debug(f"Close Msg: {response_message}")
    # Base response structure
    response = {
        'sessionState': {
            'sessionAttributes': session_attributes or {},
            'dialogAction': {'type': 'Close'},
            'intent': {'name': intent_name, 'state': fulfillment_state}
            # activeContexts managed specifically in fulfillment logic below
        },
        'messages': [{'contentType': 'PlainText', 'content': response_message}]
    }
    return response

def delegate(session_attributes, intent_name, slots):
    """Builds a Delegate response for Lex V2."""
    logger.info(f"Delegating intent: {intent_name}")
    return {
        'sessionState': {
            'sessionAttributes': session_attributes or {},
            'dialogAction': {'type': 'Delegate'},
            'intent': {'name': intent_name, 'slots': slots or {}}
        }
    }

def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message, locale):
     """Builds an ElicitSlot response for Lex V2."""
     logger.info(f"Eliciting slot: {slot_to_elicit} for intent: {intent_name}")
     response_message = message if message else get_localized_text(locale, 'default_error')
     logger.debug(f"Elicit Msg: {response_message}")
     current_slots = slots or {}
     if slot_to_elicit in current_slots:
         current_slots[slot_to_elicit] = None # Clear value before eliciting
     return {
         'sessionState': {
             'sessionAttributes': session_attributes or {},
             'dialogAction': {
                 'type': 'ElicitSlot',
                 'slotToElicit': slot_to_elicit
             },
             'intent': {
                 'name': intent_name,
                 'slots': current_slots
             }
         },
         'messages': [{'contentType': 'PlainText', 'content': response_message}]
     }

def confirm_intent(session_attributes, intent_name, slots, message, locale):
     """Builds a ConfirmIntent response for Lex V2."""
     logger.info(f"Asking Confirmation for intent: {intent_name}")
     response_message = message if message else get_localized_text(locale, 'default_error')
     logger.debug(f"Confirm Msg: {response_message}")
     return {
         'sessionState': {
             'sessionAttributes': session_attributes or {},
             'dialogAction': {'type': 'ConfirmIntent'},
             'intent': {'name': intent_name, 'slots': slots or {}}
             },
         'messages': [{'contentType': 'PlainText', 'content': response_message}]
     }


# --- Intent Fulfillment Logic ---
# Called ONLY by Fulfillment Dispatcher

def fulfill_check_balance(intent_request, locale):
    intent_name = intent_request['sessionState']['intent']['name']
    slots = intent_request['sessionState']['intent']['slots']
    session_attributes = intent_request['sessionState'].get('sessionAttributes', {})
    account_type = slots.get('accountType', {}).get('value', {}).get('interpretedValue')
    dob_value = slots.get('dateOfBirth', {}).get('value', {}).get('interpretedValue')
    if not (account_type and dob_value): logger.error(f"Fulfill CB error: Slots missing. Acct: {account_type}, DOB: {dob_value}"); return close(session_attributes, intent_name, 'Failed', get_localized_text(locale, 'default_error'), locale)
    logger.info(f"Fulfillment CheckBalance: Account='{account_type}', DOB='{dob_value}'")
    fake_balance = decimal.Decimal(random.randrange(500000, 99999999)) / decimal.Decimal(100)
    if locale == 'ja_JP':
         logger.info("Locale is ja_JP. Using direct message for balance.")
         formatted_balance_str = f"{int(fake_balance):,}"
         display_account_type_ja = JA_ACCOUNT_MAP.get(account_type, account_type)
         message = get_localized_text(locale, 'balance_confirmed_simple', account_type=display_account_type_ja, balance=formatted_balance_str)
    else:
        logger.info("Locale is not ja_JP. Using Bedrock for balance.")
        try: formatted_balance_en = f"{decimal.Decimal(fake_balance):,.2f}"
        except: formatted_balance_en = "[balance_err]"
        prompt = get_localized_text(locale, 'balance_prompt', account_type=account_type, balance=formatted_balance_en)
        generated_message = invoke_bedrock(prompt)
        if not generated_message: message = get_localized_text(locale, 'bedrock_error') + f" (Bal: {formatted_balance_en})"
        else: message = generated_message
    output_contexts = [{'name': 'contextCheckBalance', 'contextAttributes': {'dobProvided': dob_value, 'checkedAccount': account_type}, 'timeToLive': {'timeToLiveInSeconds': 90, 'turnsToLive': 5}}]
    response = close(session_attributes, intent_name, 'Fulfilled', message, locale)
    if response and 'sessionState' in response: response['sessionState']['activeContexts'] = output_contexts
    return response

def fulfill_followup_check_balance(intent_request, locale):
    intent_name = intent_request['sessionState']['intent']['name']; slots = intent_request['sessionState']['intent']['slots']; session_attributes = intent_request['sessionState'].get('sessionAttributes', {})
    account_type = slots.get('accountType', {}).get('value', {}).get('interpretedValue'); dob_value = slots.get('dateOfBirth', {}).get('value', {}).get('interpretedValue')
    if not (account_type and dob_value): logger.error(f"Fulfill Followup error: Slots missing. Acct: {account_type}, DOB: {dob_value}"); return close(session_attributes, intent_name, 'Failed', get_localized_text(locale, 'default_error'), locale)
    logger.info(f"Fulfillment Followup: Account='{account_type}', DOB='{dob_value}'")
    fake_balance = decimal.Decimal(random.randrange(10000, 5000000)) / decimal.Decimal(100)
    if locale == 'ja_JP':
         logger.info("Locale is ja_JP. Using direct template for followup.")
         formatted_balance_str = f"{int(fake_balance):,}"
         display_account_type_ja = JA_ACCOUNT_MAP.get(account_type, account_type)
         message = get_localized_text(locale, 'balance_confirmed_simple', account_type=display_account_type_ja, balance=formatted_balance_str)
    else:
        logger.info("Locale is not ja_JP. Using Bedrock for followup.")
        try: formatted_balance_en = f"{decimal.Decimal(fake_balance):,.2f}"
        except: formatted_balance_en = "[balance_err]"
        prompt = get_localized_text(locale, 'balance_prompt', account_type=account_type, balance=formatted_balance_en)
        generated_message = invoke_bedrock(prompt)
        if not generated_message: message = get_localized_text(locale, 'bedrock_error') + f" (Bal: {formatted_balance_en})"
        else: message = generated_message
    return close(session_attributes, intent_name, 'Fulfilled', message, locale)

def fulfill_make_transfer(intent_request, locale):
    # Handles post-confirmation fulfillment (Confirmed/Denied states)
    intent_name = intent_request['sessionState']['intent']['name']; slots = intent_request['sessionState']['intent']['slots']; session_attributes = intent_request['sessionState'].get('sessionAttributes', {})
    confirmation_status = intent_request['sessionState']['intent'].get('confirmationState')
    logger.info(f"Fulfillment MakeTransfer - RECEIVED State: {confirmation_status}")

    if confirmation_status == 'Denied':
        message = get_localized_text(locale, 'transfer_cancelled')
        return close(session_attributes, intent_name, 'Fulfilled', message, locale)
    elif confirmation_status == 'Confirmed':
        amount_str = slots.get('amount', {}).get('value', {}).get('interpretedValue')
        source_account_resolved = slots.get('sourceAccount', {}).get('value', {}).get('interpretedValue')
        destination_account_resolved = slots.get('destinationAccount', {}).get('value', {}).get('interpretedValue')
        if not (amount_str and source_account_resolved and destination_account_resolved): logger.error("Fulfill Transfer Confirmed missing slots!"); return close(session_attributes, intent_name, 'Failed', get_localized_text(locale, 'default_error'), locale)
        logger.info(f"Confirmed transfer sim: {amount_str} from {source_account_resolved} to {destination_account_resolved}")
        conf_num = f"SIM-{random.randint(100000, 999999)}"
        display_source = source_account_resolved; display_destination = destination_account_resolved
        if locale == 'ja_JP':
            display_source = JA_ACCOUNT_MAP.get(source_account_resolved, source_account_resolved)
            display_destination = JA_ACCOUNT_MAP.get(destination_account_resolved, destination_account_resolved)
        try: formatted_amount = f"{int(decimal.Decimal(amount_str)):,}" if locale == 'ja_JP' else f"{decimal.Decimal(amount_str):,.2f}"
        except: formatted_amount = "[amt_err]"
        message = get_localized_text(locale, 'transfer_confirmed_simple', amount=formatted_amount, source_account=display_source, destination_account=display_destination, confirmation_number=conf_num)
        return close(session_attributes, intent_name, 'Fulfilled', message, locale)
    else: # Should not be None here if dialog hook custom confirmation works
        logger.error(f"Fulfillment MakeTransfer unexpected state: {confirmation_status}")
        return close(session_attributes, intent_name, 'Failed', get_localized_text(locale, 'default_error'), locale)

def fulfill_fallback(intent_request, locale):
    # Uses Bedrock
    intent_name = intent_request['sessionState']['intent']['name']; user_input = intent_request.get('inputTranscript',''); session_attributes = intent_request['sessionState'].get('sessionAttributes',{})
    logger.info(f"Fulfillment Fallback: Input='{user_input}'");
    if not user_input: message=get_localized_text(locale,'default_error'); return close(session_attributes,intent_name,'Failed',message,locale)
    prompt = get_localized_text(locale,'fallback_prompt',user_input=user_input); gen_msg=invoke_bedrock(prompt)
    message = gen_msg if gen_msg else get_localized_text(locale,'bedrock_error'); return close(session_attributes,intent_name,'Failed',message,locale)

# --- Fulfillment Dispatcher ---
def dispatch_fulfillment(intent_request):
    """Routes fulfillment requests based on intent name."""
    intent_name = intent_request['sessionState']['intent']['name']
    locale = intent_request['bot']['localeId']
    logger.info(f"Dispatching fulfillment for intent: {intent_name}")
    if intent_name == 'CheckBalance': return fulfill_check_balance(intent_request, locale)
    if intent_name == 'FollowupCheckBalance': return fulfill_followup_check_balance(intent_request, locale)
    if intent_name == 'MakeTransfer': return fulfill_make_transfer(intent_request, locale) # Handles Confirmed/Denied only
    if intent_name == 'FallbackIntent': return fulfill_fallback(intent_request, locale)
    logger.error(f"Unhandled Fulfillment Intent: {intent_name}")
    raise ValueError(f"Unsupported fulfillment intent: {intent_name}")


# --- Lambda Entry Point ---
def lambda_handler(event, context):
    """Main Lambda handler function, routes based on invocationSource."""
    logger.info(f"Lambda Event Received: {json.dumps(event)}")
    if not bedrock_runtime: logger.critical("Bedrock client missing!"); safe_locale=event.get('bot',{}).get('localeId','en-US'); return close({},'InitError','Failed',get_localized_text(safe_locale,'default_error'), safe_locale)

    invocation_source = event.get('invocationSource')
    session_state = event.get('sessionState', {})
    intent_data = session_state.get('intent', {})
    intent_name = intent_data.get('name')
    slots = intent_data.get('slots', {})
    session_attributes = session_state.get('sessionAttributes', {})
    locale = event.get('bot', {}).get('localeId', 'en-US')
    logger.info(f"Source: {invocation_source}, Intent: {intent_name}, Locale: {locale}")

    try:
        response = None
        # --- Route based on Invocation Source ---
        if invocation_source == 'DialogCodeHook':
            logger.info(f"Processing DialogCodeHook for intent: {intent_name}") # Changed from debug
            should_delegate = True

            # --- Date Validation Logic ---
            if intent_name in ['CheckBalance', 'FollowupCheckBalance'] and slots and 'dateOfBirth' in slots and slots['dateOfBirth'] and slots['dateOfBirth'].get('value'):
                 dob_slot = slots['dateOfBirth']; original_value = dob_slot['value'].get('originalValue'); interpreted_value = dob_slot['value'].get('interpretedValue'); logger.info(f"Validating DoB: Orig='{original_value}', Interp='{interpreted_value}'")
                 is_orig_plausible = isinstance(original_value, str) and len(original_value) >= 8 and original_value.count('-') >= 2; is_interp_valid = is_valid_iso_date(interpreted_value);
                 if not is_orig_plausible or not is_interp_valid:
                      logger.warning(f"Validation Failed DoB. Re-eliciting."); elicit_message = get_localized_text(locale, 'ask_dob');
                      response = elicit_slot(session_attributes, intent_name, slots, 'dateOfBirth', elicit_message, locale)
                      should_delegate = False
                 else:
                      logger.info("Date validation Passed.") # Allows to proceed to next logic or delegate

            # --- Custom MakeTransfer Confirmation Logic ---
            # Only if not handled by date validation re-prompt
            if should_delegate and intent_name == 'MakeTransfer':
                 # Check if ALL required transfer slots have values (meaning Lex just collected the last one)
                 if all(slots.get(s) and slots[s].get('value') for s in ['amount', 'sourceAccount', 'destinationAccount']):
                      logger.info("MakeTransfer slots filled. Asking custom confirmation.")
                      amount_str = slots['amount']['value']['interpretedValue']
                      src_res = slots['sourceAccount']['value']['interpretedValue']
                      dst_res = slots['destinationAccount']['value']['interpretedValue']
                      # Map and format for the confirmation prompt
                      display_source = src_res; display_destination = dst_res
                      if locale == 'ja_JP':
                          display_source = JA_ACCOUNT_MAP.get(src_res, src_res); display_destination = JA_ACCOUNT_MAP.get(dst_res, dst_res);
                      try: amount_fmt = f"{int(decimal.Decimal(amount_str)):,}" if locale == 'ja_JP' else f"{decimal.Decimal(amount_str):,.2f}";
                      except: amount_fmt = amount_str # Fallback
                      conf_msg = get_localized_text(locale, 'transfer_confirmation_custom', amount=amount_fmt, source_account=display_source, destination_account=display_destination)
                      response = confirm_intent(session_attributes, intent_name, slots, conf_msg, locale)
                      should_delegate = False # We handled this by asking for confirmation

            # If no specific action (validation failure or confirmation prompt) was taken by this point, delegate.
            if should_delegate:
                logger.info(f"No specific Dialog action taken for {intent_name}. Delegating.")
                response = delegate(session_attributes, intent_name, slots)

        elif invocation_source == 'FulfillmentCodeHook':
            logger.info(f"Processing FulfillmentCodeHook for intent: {intent_name}")
            response = dispatch_fulfillment(event) # Call the fulfillment dispatcher

        else: # Handle unknown invocation source
            logger.error(f"Unknown invocationSource: {invocation_source}")
            response = close(session_attributes, intent_name or 'ErrorIntent', 'Failed', get_localized_text(locale, 'default_error'), locale)

    except Exception as e: # Catch all other errors
        logger.error(f"Exception during {invocation_source or 'Unknown'} processing for intent {intent_name}: {e}", exc_info=True)
        response = close(session_attributes, intent_name or 'ErrorIntent', 'Failed', get_localized_text(locale, 'default_error'), locale)

    logger.info(f"Lambda Final Response Sent: {json.dumps(response)}")
    return response
# --- End Complete FINAL Lambda Code ---

```

## Phase 4: Connect Lex to Lambda & Final Build

**Step 4.1: Add Lex Invoke Lambda Permission to Lex Role**

1.  Go back to **IAM -> Roles**.
2.  Click on `LexV2BotRole-ProBilingual-Tokyo`.
3.  **Permissions** tab -> **Add permissions** -> **Create inline policy** -> **JSON**.
4.  **PASTE** (REPLACE `<YOUR_ACCOUNT_ID>`):
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [ { "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": "arn:aws:lambda:ap-northeast-1:<YOUR_ACCOUNT_ID>:function:ProBilingualBankerHandler-Tokyo" } ]
    }
    ```
5.  Review -> Name: `InvokeProBilingualLambdaPolicy` -> **Create policy**.

**Step 4.2: Connect Intents & Code Hooks to Lambda**

1.  Go back to **Amazon Lex V2** -> `ProBilingualBankerBot-Tokyo`.
2.  **For `CheckBalance` and `FollowupCheckBalance` intents:**
    *   Select intent -> Scroll to **Code hooks**.
    *   Check **"Use a Lambda function for initialization and validation"**.
    *   Select `ProBilingualBankerHandler-Tokyo`.
    *   Click **Save intent**.
3.  **For `CheckBalance`, `FollowupCheckBalance`, `MakeTransfer`, and `FallbackIntent`:**
    *   Select intent -> Scroll to **Fulfillment**.
    *   Ensure "Active" is ON. Expand "On successful fulfillment" -> Advanced options.
    *   Check **"Use a Lambda function for fulfillment"**.
    *   Select `ProBilingualBankerHandler-Tokyo`.
    *   Click "Update options" -> **Save intent**.

**Step 4.3: Final Backend Build**

1.  Click **Build**. Wait 1-2 minutes.

---

## Phase 5: Backend Testing (Lex Console)

1.  Click **Test**.
2.  **Test English (`en-US`):**
    *   `check checking balance` -> (Bot asks DOB) -> `1990-01-01` -> Expect Bedrock EN balance.
    *   `how about savings` -> Expect Bedrock EN balance (no DOB prompt).
    *   `transfer 100 from checking to savings` -> (Bot asks custom EN confirmation) -> `yes` -> Expect EN template transfer success.
    *   `what is the weather` -> Expect Bedrock EN fallback.
3.  **Test Japanese (`ja-JP`):**
    *   Switch language to Japanese.
    *   `普通預金の残高` -> (Bot asks DOB) -> `1990-01-01` -> Expect Direct JA balance.
    *   `貯蓄預金は` -> Expect Direct JA balance (no DOB prompt).
    *   `普通預金から貯蓄預金へ3000円振替` -> (Bot asks custom JA confirmation) -> `はい` -> Expect JA template transfer success.
    *   `今日の天気は` -> Expect Bedrock JA fallback.
4.  **Monitor CloudWatch Logs** for `ProBilingualBankerHandler-Tokyo` for details and errors.

---

## Phase 6: Frontend Chat Interface (HTML/CSS/JS - Simplified)

*   This part creates a basic web page to chat with your bot.

**Step 6.1: Create Cognito Identity Pool**

1.  Go to **AWS Cognito** (in `ap-northeast-1`).
2.  **Manage Identity Pools** -> **Create new identity pool**.
3.  Name: `ProBilingualChatbotPool-Tokyo`.
4.  Check **Enable access to unauthenticated identities**. Click **Create Pool**. Click **Allow**.
5.  **COPY the `IdentityPoolId`** (e.g., `ap-northeast-1:xxxxxxxx-...`).

**Step 6.2: Grant Cognito Role Lex Permission**

1.  Go to **IAM -> Roles**. Search `Cognito_ProBilingualChatbotPool-TokyoUnauth_Role`. Click it.
2.  **Add permissions** -> **Create inline policy** -> **JSON**.
3.  **PASTE** (REPLACE `<ACCOUNT_ID>`, `<BOT_ID>`):
    ```json
    { "Version": "2012-10-17", "Statement": [ { "Effect": "Allow", "Action": "lex:RecognizeText", "Resource": "arn:aws:lex:ap-northeast-1:<ACCOUNT_ID>:bot-alias/<BOT_ID>/TSTALIASID" } ] }
    ```
4.  Review -> Name: `LexUnauthAccessPolicy-Tokyo` -> **Create policy**.

**Step 6.3: Create Frontend Files in `MyProBilingualChatbot` Folder**

1.  **`chat.html`:**
    ```html
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bilingual Banker Bot</title><link rel="stylesheet" href="chat.css"></head>
    <body><h1>Chat with ProBilingual Banker Bot</h1>
        <div id="language-selector">
             <label><input type="radio" name="language" value="en-US" checked> English</label>
             <label><input type="radio" name="language" value="ja_JP"> 日本語</label> <!-- Use ja_JP -->
        </div>
        <div id="chat-container"><div id="chat-header"><h2 id="chat-title">Banker Bot</h2></div>
            <div id="chat-messages"></div>
            <div id="chat-input-area"><input type="text" id="user-input" placeholder="Type..."><button id="send-button">Send</button></div>
             <div id="loading-indicator" style="display: none;">Thinking...</div><div id="error-message" style="color: red;"></div>
        </div>
        <script src="https://unpkg.com/@aws-sdk/client-lex-runtime-v2/dist-es/index.js"></script>
        <script src="https://unpkg.com/@aws-sdk/credential-provider-cognito-identity/dist-es/index.js"></script>
        <script type="module" src="chat.js"></script></body></html>
    ```
2.  **`chat.css`:** (Use CSS from previous guide - Step 4.3, or customize).
3.  **`chat.js`:**
    ```javascript
    // chat.js - Bilingual
    const { LexRuntimeV2Client, RecognizeTextCommand } = LexRuntimeV2Client;
    const { fromCognitoIdentity } = credentialProviderCognitoIdentity;

    const AWS_REGION = "ap-northeast-1";
    const BOT_ID = "YOUR_K6JWC9IBRK_BOT_ID"; // REPLACE with your Lex Bot ID
    const BOT_ALIAS_ID = "TSTALIASID"; // Standard Test Alias ID
    const COGNITO_IDENTITY_POOL_ID = "YOUR_COGNITO_IDENTITY_POOL_ID_HERE"; // REPLACE

    const UI_TEXT = { /* ... (Keep full UI_TEXT from previous guide) ... */
        'en-US': { title: "Banker Bot", inputPlaceholder: "Type...", sendButton: "Send", loading: "Thinking...", initialMsg: "Hello! How can I help?" },
        'ja_JP': { title: "バンカーボット", inputPlaceholder: "入力...", sendButton: "送信", loading: "考え中...", initialMsg: "こんにちは！ご用件は何ですか？" }
    };
    const ERROR_TEXT = { /* ... (Keep as before) ... */
         'en-US': { errorConnect: "Connection error.", errorConfig: "Config error (F12)." },
         'ja_JP': { errorConnect: "接続エラー。", errorConfig: "設定エラー（F12）。" }
    };

    let lexClient;
    try {
        const credentials = fromCognitoIdentity({ identityPoolId: COGNITO_IDENTITY_POOL_ID, clientConfig: { region: AWS_REGION }});
        lexClient = new LexRuntimeV2Client({ region: AWS_REGION, credentials });
    } catch (e) { console.error("SDK/Cognito Init Error:", e); document.getElementById('error-message').textContent="SDK Init Failed.";}

    const messagesContainer = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');
    const languageRadios = document.querySelectorAll('input[name="language"]');
    const chatTitle = document.getElementById('chat-title');

    let sessionId = 'webchat-session-' + Date.now();
    let currentLocale = 'en-US';

    function updateUILanguage(locale) { /* ... (Keep as before) ... */
        const texts = UI_TEXT[locale] || UI_TEXT['en-US']; chatTitle.textContent=texts.title; userInput.placeholder=texts.inputPlaceholder; sendButton.textContent=texts.sendButton; loadingIndicator.textContent=texts.loading;
    }
    function addMessage(text, sender) { /* ... (Keep as before) ... */
        errorMessageDiv.textContent=''; const div=document.createElement('div'); div.classList.add('message', sender==='user'?'user-message':'bot-message'); div.textContent=text; messagesContainer.appendChild(div); messagesContainer.scrollTop=messagesContainer.scrollHeight;
    }

    async function sendMessage() {
        if (!lexClient) {errorMessageDiv.textContent=ERROR_TEXT[currentLocale]?.errorConfig; return;}
        const text = userInput.value.trim(); if (!text) return;
        addMessage(text, 'user'); userInput.value = ''; loadingIndicator.style.display = 'block'; sendButton.disabled = true; userInput.disabled = true;
        const params = { botId: BOT_ID, botAliasId: BOT_ALIAS_ID, localeId: currentLocale, sessionId: sessionId, text: text };
        try {
            const command = new RecognizeTextCommand(params); const response = await lexClient.send(command);
            if (response.messages && response.messages.length > 0) { response.messages.forEach(msg => msg.content && addMessage(msg.content, 'bot'));}
            else { console.log("Lex no messages, state:", response.sessionState?.dialogAction?.type); }
        } catch (error) { console.error("Lex call error:", error); addMessage(ERROR_TEXT[currentLocale]?.errorConnect, 'bot'); errorMessageDiv.textContent=ERROR_TEXT[currentLocale]?.errorConfig;}
        finally { loadingIndicator.style.display = 'none'; sendButton.disabled = false; userInput.disabled = false; userInput.focus();}
    }
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
    languageRadios.forEach(radio => { radio.addEventListener('change', (e) => { currentLocale = e.target.value; updateUILanguage(currentLocale); messagesContainer.innerHTML=''; addMessage(UI_TEXT[currentLocale]?.initialMsg, 'bot'); sessionId = 'webchat-session-' + Date.now(); }); });
    updateUILanguage(currentLocale); addMessage(UI_TEXT[currentLocale].initialMsg, 'bot');
    ```
    *   **REPLACE `YOUR_K6JWC9IBRK_BOT_ID` and `YOUR_COGNITO_IDENTITY_POOL_ID_HERE` with your actual IDs.**
    *   Fill in the `UI_TEXT` and `ERROR_TEXT` dictionaries more completely if desired.

---
---
## Phase 7: Running & Testing Full Application

1.  Open the `MyProBilingualChatbot` folder.
2.  **Double-click `chat.html`**. It should open in your browser.
3.  Test interactions in both English and Japanese.

---
---
## Phase 8: Cleanup - MANDATORY!

*   **DELETE ALL RESOURCES in `ap-northeast-1`**
    1.  **Lex Bot:** `ProBilingualBankerBot-Tokyo`.
    2.  **Lambda Function:** `ProBilingualBankerHandler-Tokyo`.
    3.  **Cognito Identity Pool:** `ProBilingualChatbotPool-Tokyo` (and its IAM Roles).
    4.  **IAM Roles:** `LexV2BotRole-ProBilingual-Tokyo`, `LambdaExecutionRole-ProBilingual-Tokyo`.
    5.  **CloudWatch Log Groups** associated with the Lambda function.

---
