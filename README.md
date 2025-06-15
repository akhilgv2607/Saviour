# Saviour
This repository is for undertanding how git and github works 
Saviour is a Streamlit-based retrieval-augmented generation (RAG) assistant that helps students interactively query and summarize academic course materials. It supports a variety of file formats (PDF, DOCX, TXT, audio, video, and images via OCR) and uses LangChain with Groq’s LLaMA-3 model under the hood.

🚀 Features

Multi-format Input: Upload course resources in PDF, DOCX, TXT, MP3/WAV audio, MP4/MOV video, and JPG/PNG images.

OCR for Images: Extract text from uploaded images using Tesseract OCR.

Vector Store & Retriever: Leverage FAISS and Sentence-Transformers embeddings to create a searchable vector store of your documents.

Conversational RAG: Ask follow-up questions—memory buffer preserves context across turns.

Summarization Mode: Ask for summaries of uploaded materials (triggered by keywords like "summarize").

ChatGPT-Like UI: Clean, chat-style interface using Streamlit’s st.chat_message and st.chat_input.

Automatic Input Clearing: Input box resets after each query for a seamless chat experience.

📦 Installation

Clone the repository

git clone https://github.com/<your-username>/saviour-rag-assistant.git
cd saviour-rag-assistant

Create & activate a virtual environment

python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate     # Windows

Install dependencies

pip install -r requirements.txt

Install Tesseract OCR

Windows: Download and install from the Tesseract repo.

macOS: brew install tesseract

Ubuntu/Debian: sudo apt-get install tesseract-ocr

Add Tesseract to your PATH (if needed) or set pytesseract.pytesseract.tesseract_cmd in code.

Set your environment variable

export GROQ_API_KEY=<your_groq_api_key>    # macOS/Linux
set GROQ_API_KEY=<your_groq_api_key>       # Windows

🛠️ Usage

streamlit run app.py

Upload your course material using the sidebar uploader.

Type your query in the chat input (e.g. "What are the main topics in Chapter 2?").

Click Send or press the built-in button.

Review the assistant’s answer and follow up with new questions.

Tip: Include "summarize" in your prompt to get a summary of the entire document.

🔧 Project Structure
── rag_app.py          # Main Streamlit application
├── utils/             # Helper functions for loading PDF, audio, video
├── requirements.txt   # Python package dependencies
└── README.md          # This file

Made by Akhil
