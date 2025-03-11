

This is a comprehensive overview of **major AI APIs**, their setup steps, key features, and use cases across text, vision, audio, and multimodal tasks:

---

### **1. OpenAI**
- **Key APIs**: GPT-4, GPT-4o, DALL·E 3, Whisper, Embeddings, Moderation.
- **Setup**:
  ```bash
  pip install openai
  ```
  - Get an API key from [OpenAI Platform](https://platform.openai.com/).
  ```python
  from openai import OpenAI
  client = OpenAI(api_key="YOUR_KEY")
  ```
- **Features**:
  - **Chat Completions**: Multi-turn conversations, JSON mode, function calling.
    ```python
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "Explain quantum computing"}]
    )
    ```
  - **Vision**: GPT-4o processes images/text in parallel.
  - **TTS/STT**: Convert text to speech (`tts-1`) or transcribe audio (`Whisper`).
  - **Fine-tuning**: Customize models with proprietary data.
- **Pricing**: Pay-as-you-go (per 1M tokens).
- **Use Cases**: Chatbots, code generation, document analysis, synthetic data.

---

### **2. Google AI (Gemini & Vertex AI)**
- **Key APIs**: Gemini Pro (text), Gemini Vision (multimodal), Vertex AI (custom models).
- **Setup**:
  ```bash
  pip install google-generativeai
  ```
  - Get API key from [Google AI Studio](https://makersuite.google.com/).
  ```python
  import google.generativeai as genai
  genai.configure(api_key="YOUR_KEY")
  ```
- **Features**:
  - **Multimodal Inputs**: Mix text, images, and video in prompts.
    ```python
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(["Explain this image", image])
    ```
  - **Vertex AI**: Train/deploy models on GCP with AutoML or custom containers.
  - **Embeddings**: Text/visual similarity via `text-embedding-004`.
- **Pricing**: Tiered based on model and usage (e.g., $0.000125/1k tokens for Gemini Pro).
- **Use Cases**: Multimodal search, document parsing, enterprise AI agents.

---

### **3. AWS Bedrock**
- **Key Models**: Claude 3 (Anthropic), Llama 2, Titan, Jurassic-2.
- **Setup**:
  - Enable Bedrock in AWS Console (region-specific).
  - Use `boto3`:
  ```python
  import boto3
  bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
  ```
- **Features**:
  - **Serverless Access**: No infrastructure management.
  - **Claude 3 Opus/Sonnet**: State-of-the-art reasoning and vision.
    ```python
    response = bedrock.invoke_model(
        body=json.dumps({"prompt": "Hello!", "max_tokens": 100}),
        modelId="anthropic.claude-v2"
    )
    ```
  - **Guardrails**: Content moderation filters.
- **Pricing**: Per 1M input/output tokens (e.g., Claude 3 Opus: $15/1M input tokens).
- **Use Cases**: Enterprise-scale apps, secure AI workflows.

---

### **4. Anthropic (Claude)**
- **Key Models**: Claude 3 Haiku, Sonnet, Opus.
- **Setup**:
  ```bash
  pip install anthropic
  ```
  ```python
  import anthropic
  client = anthropic.Anthropic(api_key="YOUR_KEY")
  ```
- **Features**:
  - **200k Context Window**: Process long documents (e.g., legal contracts).
  - **Vision**: Analyze charts, diagrams, and screenshots.
    ```python
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Analyze this image", "image": "base64_data"}]
    )
    ```
  - **Constitutional AI**: Built-in ethical safeguards.
- **Pricing**: Claude 3 Opus ($15/1M input tokens), Haiku ($0.25/1M input tokens).
- **Use Cases**: Legal analysis, research, code review.

---

### **5. Cohere**
- **Key APIs**: Command-R+, Embed, Generate, Classify.
- **Setup**:
  ```bash
  pip install cohere
  ```
  ```python
  import cohere
  co = cohere.Client("YOUR_KEY")
  ```
- **Features**:
  - **RAG Optimization**: Built-in retrieval-augmented generation.
    ```python
    response = co.chat(
        message="What is quantum entanglement?",
        documents=[{"title": "Physics 101", "text": "..."}]
    )
    ```
  - **Multilingual**: Supports 100+ languages.
  - **Embeddings**: Semantic search with `embed-english-v3.0`.
- **Pricing**: Free tier + usage-based (e.g., Command-R+ at $0.50/1M tokens).
- **Use Cases**: Enterprise search, multilingual chatbots.

---

### **6. Meta (Llama 2/3)**
- **Key Models**: Llama 3 (8B/70B), Code Llama, Llama Guard.
- **Setup**:
  - Access via Hugging Face or local deployment.
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
  ```
- **Features**:
  - **Open Weights**: Commercially usable (request access via Meta).
  - **Customization**: Fine-tune with LoRA/QLoRA.
  - **Code Generation**: Specialized CodeLlama-34b variants.
- **Use Cases**: On-premise LLMs, research, cost-sensitive deployments.

---

### **7. Hugging Face Inference API**
- **Key Models**: 200k+ community models (Mixtral, Falcon, Stable Diffusion).
- **Setup**:
  ```bash
  pip install huggingface_hub
  ```
  ```python
  from huggingface_hub import InferenceClient
  client = InferenceClient(token="YOUR_TOKEN")
  ```
- **Features**:
  - **Open Source Focus**: Run models like `mistral-7b-instruct`.
    ```python
    response = client.text_generation(
        prompt="Explain AI in one sentence:", model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    ```
  - **Endpoints**: Deploy private models as scalable APIs.
- **Pricing**: Free for small tasks; pay for dedicated endpoints.
- **Use Cases**: Experimentation, open-source LLM deployment.

---

### **8. Perplexity AI**
- **Key APIs**: Online LLMs (pplx-7b-online, mistral-7b-instruct).
- **Setup**:
  ```python
  pip install perplexity-api
  ```
  ```python
  from perplexity import Perplexity
  pp = Perplexity(api_key="YOUR_KEY")
  ```
- **Features**:
  - **Real-Time Web Access**: Ground responses with live data.
    ```python
    response = pp.query("Latest news on SpaceX Starship")
    ```
  - **Low Latency**: Optimized for speed.
- **Pricing**: $0.20/1k tokens (pay-as-you-go).
- **Use Cases**: Research assistants, news aggregation.

---

### **9. Mistral AI**
- **Key Models**: Mixtral 8x7B, Mistral 7B, Codestral (code).
- **Setup**:
  - Use via API or self-host:
  ```python
  from mistralai.client import MistralClient
  client = MistralClient(api_key="YOUR_KEY")
  ```
- **Features**:
  - **Sparse Mixture-of-Experts (MoE)**: Efficient inference.
  - **Function Calling**: Structured JSON output.
  ```python
  response = client.chat(
      model="mistral-large-latest",
      messages=[{"role": "user", "content": "Write a Python function for Fibonacci"}]
  )
  ```
- **Pricing**: Mistral Large ($8/1M input tokens).
- **Use Cases**: Code generation, low-latency apps.

---

### **10. Microsoft Azure AI**
- **Key APIs**: Azure OpenAI (GPT-4, DALL·E), Cognitive Services (Vision, Speech).
- **Setup**:
  - Use Azure Portal to deploy models.
  ```python
  from openai import AzureOpenAI
  client = AzureOpenAI(
      api_key="YOUR_KEY",
      api_version="2023-12-01-preview",
      azure_endpoint="https://YOUR_RESOURCE.openai.azure.com"
  )
  ```
- **Features**:
  - **Enterprise Security**: SOC 2 compliance, private networking.
  - **Custom Models**: Fine-tune GPT-3.5/4 with Azure Machine Learning.
- **Pricing**: Similar to OpenAI, with Azure credits.
- **Use Cases**: Healthcare, finance, regulated industries.

---

### **Comparison Table**
| **API**       | **Best For**           | **Cost (per 1M tokens)** | **Key Strength**               |
|---------------|------------------------|--------------------------|---------------------------------|
| OpenAI        | General-purpose tasks  | $10 (GPT-4 Turbo)        | Versatility, vision support    |
| Claude 3      | Long-context reasoning | $15 (Opus)               | Ethics, 200k tokens            |
| Gemini        | Multimodal apps        | $0.000125 (Gemini Pro)   | Google ecosystem integration   |
| AWS Bedrock   | Enterprise scalability | Varies by model          | Serverless, multi-model access |
| Mistral AI    | Cost efficiency        | $0.50 (Mistral 7B)      | Open-source MoE models         |
| Cohere        | RAG workflows          | $0.50 (Command-R+)       | Retrieval-augmented generation |

---

### **Choosing the Right API**
- **Budget**: Mistral/Cohere for cost-sensitive projects; OpenAI/Claude for high performance.
- **Use Case**: 
  - **Multimodal**: GPT-4o, Gemini.
  - **Long Context**: Claude 3, GPT-4 Turbo.
  - **Self-Hosted**: Llama 3, Hugging Face.
- **Compliance**: Azure OpenAI or AWS Bedrock for regulated industries.

---

Leverage these APIs to build chatbots, automate workflows, analyze unstructured data, or create multimodal apps. Always test with **proof-of-concepts** to compare performance and cost!
