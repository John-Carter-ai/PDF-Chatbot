

import streamlit as st
import os
os.environ["CHROMA_DB_IMPL"] = "duckdb"
import asyncio

# LangChain & Google GenAI imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#import compat  # patches SQLite before Chroma loads

# 🛠️ Ensure event loop for async client (required in Streamlit thread)
def ensure_event_loop():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

# Set up Google service account credentials
api_key = st.secrets["GOOGLE_API_KEY"] 

#  Load and split PDF
pdf_path = r"my_paper.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)

#  Embed and store vectors
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(split_docs, embedding=embedding)
retriever = vectorstore.as_retriever(search_type='similarity')

#  Set up Gemini LLM and prompt
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

system_prompt = "You are my personal assistant to talk with PDF. {context}"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

#  Build RAG chain
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 🖼 Streamlit UI
st.title("📚 PDF Chat Assistant")
query = st.chat_input("Ask me anything about the PDF:")

if query:
    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])




