import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")

@st.cache_resource()
def load_summarizer():
    return pipeline("text2text-generation", model="facebook/bart-large-cnn")

def text_summary(text):
    summarizer = load_summarizer()
    # Split long text into chunks of 1000 chars
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = " ".join([summarizer(chunk, max_length=150, min_length=30)[0]['generated_text'] for chunk in chunks])
    return summary

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join([p.get_text() for p in paragraphs])

st.title("Text Summarizer")
choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    input_text = st.text_area("Enter your text here")
    if st.button("Summarize Text"):
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("**Your Input Text**")
            st.info(input_text)
        with col2:
            st.markdown("**Summary Result**")
            result = text_summary(input_text)
            st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    upload_option = st.radio("Choose upload method", ("Upload PDF", "Input URL"))
    if upload_option == "Upload PDF":
        input_file = st.file_uploader("Upload your document here", type=['pdf'])
        if input_file and st.button("Summarize Document"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf(input_file)
                st.markdown("**Extracted Text:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                st.success(text_summary(extracted_text))
    else:
        input_url = st.text_input("Enter the URL")
        if input_url and st.button("Summarize Document"):
            text_from_url = fetch_text_from_url(input_url)
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Extracted Text:**")
                st.info(text_from_url)
            with col2:
                st.markdown("**Summary Result**")
                st.success(text_summary(text_from_url))
