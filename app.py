# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

# Initialize OpenAI API key
openai_api_key = 'sk-proj-b3MjzhCPmezBx8NQyhyxT3BlbkFJO8r5FQeWIdeZYOAinTpA'

# Hardcoded PDF path
pdf_path = "visa-requirements-dataset-_1_.pdf"

# Function to read PDF content using PyPDF2
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load and process the PDF document
@st.cache_data
def load_pdf(file):
    try:
        text = read_pdf(file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        documents = [{"text": t, "metadata": {"page_content": t}} for t in texts]  # Updated line
        return documents
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

# Initialize Chroma vector store
def init_chroma(texts):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(texts, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing Chroma: {e}")
        return None

# Function to perform web search on Visafy website
def search_visafy_website(query):
    try:
        url = "https://visafy.org"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        st.error(f"Error searching Visafy website: {e}")
        return ""

st.title("Visafy AI Assistant Chatbot")

# Load the PDF document
texts = load_pdf(pdf_path)

if texts:
    vectorstore = init_chroma(texts)
    if vectorstore:
        qa = RetrievalQA.from_chain_type(llm=OpenAI(api_key=openai_api_key), chain_type="stuff", retriever=vectorstore.as_retriever())

        query = st.text_input("Ask a question about visa applications:")
        if query:
            try:
                result = qa.run(query)
                if not result:
                    st.write("Searching on Visafy website...")
                    web_result = search_visafy_website(query)
                    st.write(web_result)
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"Error processing query: {e}")
