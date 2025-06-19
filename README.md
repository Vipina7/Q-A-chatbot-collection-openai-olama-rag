# 🧠 Multi-Agent Q&A Chatbots and RAG with LLaMA3, OLlama, OpenAI & Groq

This repository contains three powerful GenAI applications:

1. **End-to-End Q&A Chatbot with OpenAI**  
2. **End-to-End Q&A Chatbot with OLlama (local model)**  
3. **RAG-based Document Q&A using Groq API and LLaMA3**

Built using **Streamlit** and **LangChain**, these apps showcase both general-purpose Q&A and retrieval-augmented generation from documents.

---

## 📁 Project Structure

```bash
├── OpenAI.py                  # Q&A Chatbot using OpenAI (GPT-4/4-turbo/4o)
├── Ollama.py                  # Q&A Chatbot using OLlama (Gemma, DeepSeek, etc.)
├── RAGDocument_QA_GROQ.py     # Document-based Q&A using Groq API + LLaMA3 + FAISS
├── .env                       # Environment variables for API keys (not committed)
├── repository/                # Folder containing PDF documents for RAG
├── .gitignore                 # Git ignore configuration
└── README.md                  # This file
```

---

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment**
   ```bash
   conda create -p venv python==3.10
   conda activate venv/  # Activate virtual environment
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```env
   LANGCHAIN_API_KEY=your_langchain_api_key
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   HF_TOKEN=your_huggingface_token
   ```

5. **Add documents for RAG**
   - Place your `.pdf` files inside the `repository/` directory.

---

## 🚀 How to Run

### 1. OpenAI Chatbot
```bash
streamlit run OpenAI.py
```

### 2. OLlama Chatbot
```bash
streamlit run Ollama.py
```

> ✅ Make sure you have OLlama installed and that models like `gemma` or `deepseek` are available locally.

### 3. RAG with Groq + LLaMA3
```bash
streamlit run RAGDocument_QA_GROQ.py
```

Click the "Document Embedding" button first to index your PDFs.

---

## 📌 Features

- 🔓 OpenAI + Local model (OLlama) flexibility  
- 📚 PDF document Q&A via RAG pipeline  
- ⚡ Uses **FAISS** vector store for fast semantic search  
- 🧠 LangChain for chaining prompts and managing flows  
- 🔥 Streamlit UI with sliders for temperature/token control  

---

## 📥 Requirements

Create a `requirements.txt` with:

```txt
langchain
openai
langchain_community
langchain-openai
langchain_core
langchain_huggingface
langchain_groq
faiss-cpu
python-dotenv
sentence_transformers
pypdf
streamlit
```

> ✅ You may also need `torch`, `transformers`, or `llama-cpp` depending on OLlama setup.

---
