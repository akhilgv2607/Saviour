import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq
from langchain.document_loaders import PyPDFLoader
from utils.loader import load_pdf, load_audio, load_video
from PIL import Image
import pytesseract
from docx import Document

# ----------------- Init LLM -------------------
def init_llm():
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY", "Your_API_Key")
    )

llm = init_llm()

# ----------------- Page Config -------------------
st.set_page_config(page_title="Saviour")
st.title("Saviour ✝️")
st.write("Hi!")

# ----------------- Helper: Vector Store -------------------
@st.cache_resource
def create_vectorstore_from_text(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([texts])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, docs

@st.cache_resource
def create_vectorstore_from_pdf(file):
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(file.read())
    loader = PyPDFLoader("temp_uploaded.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, docs

@st.cache_resource
def create_vectorstore_from_docx(file):
    # Read .docx and extract text
    with open("temp_uploaded.docx", "wb") as f:
        f.write(file.read())
    doc = Document("temp_uploaded.docx")
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return create_vectorstore_from_text(full_text)


def summarize_docs(docs):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

# ----------------- Session State -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

# ----------------- Display Chat History -------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# ----------------- File Upload & User Input -------------------
col_file, col_input = st.columns([1, 2])
with col_file:
    uploaded_file = st.file_uploader(
        "Upload a file"
    )
with col_input:
    user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            if uploaded_file:
                mime = uploaded_file.type
                # PDF
                if mime == "application/pdf":
                    vectorstore, docs = create_vectorstore_from_pdf(uploaded_file)
                # DOCX
                elif uploaded_file.name.lower().endswith(".docx"):
                    vectorstore, docs = create_vectorstore_from_docx(uploaded_file)
                # Text
                elif mime == "text/plain":
                    raw = uploaded_file.read().decode("utf-8")
                    vectorstore, docs = create_vectorstore_from_text(raw)
                # Audio
                elif "audio" in mime:
                    audio_text = load_audio(uploaded_file)
                    vectorstore, docs = create_vectorstore_from_text(audio_text)
                # Video
                elif "video" in mime:
                    video_text = load_video(uploaded_file)
                    vectorstore, docs = create_vectorstore_from_text(video_text)
                # Image (OCR)
                elif "image" in mime:
                    img = Image.open(uploaded_file)
                    ocr_text = pytesseract.image_to_string(img)
                    vectorstore, docs = create_vectorstore_from_text(ocr_text)
                else:
                    st.error("Unsupported file type! Please upload PDF, DOCX, TXT, Image, Audio, or Video.")
                    st.stop()

                retriever = vectorstore.as_retriever()
                if "summarize" in user_input.lower():
                    summary = summarize_docs(docs)
                    st.session_state.chat_history.append(("assistant", summary))
                    st.write(summary)
                else:
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        return_source_documents=True,
                        output_key="answer",
                    )
                    result = qa_chain.invoke({"question": user_input})
                    answer = result["answer"]
                    st.session_state.chat_history.append(("assistant", answer))
                    st.write(answer)
            else:
                # No upload; plain LLM
                result = llm.invoke(user_input)
                answer = result.content
                st.session_state.chat_history.append(("assistant", answer))
                st.write(answer)
