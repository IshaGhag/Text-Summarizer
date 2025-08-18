import streamlit as st
from txtai.pipeline import Summary
from PyPDF2 import PdfReader
import requests

st.set_page_config(layout="wide")

@st.cache_data()
def text_summary(text, maxlength=None):
    # Create summary instance
    summary = Summary()
    result = summary(text)
    return result

def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            page = reader.pages[0]
            text = page.extract_text()
        return text
    except Exception as e:
        print("Error extracting text from PDF:", e)
        return None

def fetch_text_from_url(url):
    # Fetch text from URL
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
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
        if input_file is not None:
            if st.button("Summarize Document"):
                with open("doc_file.pdf", "wb") as f:
                    f.write(input_file.getbuffer())
                col1, col2 = st.columns([1,1])
                with col1:
                    st.info("File uploaded successfully")
                    extracted_text = extract_text_from_pdf("doc_file.pdf")
                    print("Extracted Text:", extracted_text)  # Debugging statement
                    st.markdown("**Extracted Text is Below:**")
                    st.info(extracted_text)
                with col2:
                    st.markdown("**Summary Result**")
                    doc_summary = text_summary(extracted_text)
                    st.success(doc_summary)
    else: # Input URL
        input_url = st.text_input("Enter the URL of the document")
        if input_url:
            if st.button("Summarize Document"):
                text_from_url = fetch_text_from_url(input_url)
                if text_from_url:
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.info("Text fetched successfully")
                        st.markdown("**Extracted Text is Below:**")
                        st.info(text_from_url)
                    with col2:
                        st.markdown("**Summary Result**")
                        doc_summary = text_summary(text_from_url)
                        st.success(doc_summary)
                else:
                    st.error("Failed to fetch text from URL. Please check if the URL is correct or accessible.")
