# 🦊 GitLab Handbook Assistant

A conversational RAG chatbot built to answer questions based on the GitLab Handbook using LangChain (LCEL), ChromaDB, and Gemini 1.5 Flash. Built for the Joveo Round 2 Assessment.

## Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Add API Keys:**
Create a `.env` file in the root directory and add:
```env
GOOGLE_API_KEY="your_gemini_api_key_here"
```

## Run Locally

Launch the Streamlit interface:
```bash
streamlit run app.py
```
