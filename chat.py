import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import asyncio
load_dotenv()
os.environ["GROQ_API_KEY"]="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"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize as an empty list
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
def load_documents(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_chunks(text):
    textsplitter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=textsplitter.split_text(text)
    return chunks
def get_vectore_store(chunks):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
def get_conversation_chain(vector_store):
    llm=ChatGroq()
    memory = ConversationBufferMemory(
    llm=llm,
    output_key="answer",
    memory_key="chat_history",
    return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory)
    return conversation_chain
async def chat_bot():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about the uploaded PDF."}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt=st.chat_input("Ask your question")
    if prompt :
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
        result=st.session_state.conversation_chain.invoke(prompt)
        st.chat_message("assistant").write(result["answer"])
        st.session_state.messages.append({"role":"assistant","content":result["answer"]})
st.set_page_config(page_title="Chat with PDFS",page_icon="")
st.header("CHAT WITH YOUR PDF")
uploaded_files=st.file_uploader("Upload your PDF",accept_multiple_files=True)
async def main():
    if uploaded_files:
        text_content = load_documents(uploaded_files)
        text_chunks=get_chunks(text_content)
        vector_store=get_vectore_store(text_chunks)
        st.session_state.conversation_chain=get_conversation_chain(vector_store)
        await chat_bot()
if __name__=="__main__":
     asyncio.run(main())
