import streamlit as st
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

st.set_page_config(layout="wide")

def text_summary(text, sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join([str(s) for s in summary])

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages])

def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join([p.get_text() for p in soup.find_all("p")])

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
            st.success(text_summary(input_text))

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    upload_option = st.radio("Choose upload method", ("Upload PDF", "Input URL"))
    if upload_option == "Upload PDF":
        input_file = st.file_uploader("Upload your document here", type=['pdf'])
        if input_file and st.button("Summarize Document"):
            col1, col2 = st.columns([1,1])
            with col1:
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
