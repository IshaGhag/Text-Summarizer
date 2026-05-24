import streamlit as st
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
import re
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #f9f7f4; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background: #1c1c1e !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] small { color: #e5e5e7 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: #2c2c2e !important; border-color: #3a3a3c !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span { color: #ffffff !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #2c2c2e !important; border: 1px solid #3a3a3c !important;
    color: #ffffff !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder { color: #8e8e93 !important; }
[data-testid="stSidebar"] hr { border-color: #3a3a3c !important; }

/* ── BLOCK CONTAINER ── */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 0 !important;
    max-width: 1200px;
}

/* ── PAGE HEADER ── */
.page-header {
    margin-bottom: 1.75rem;
    border-bottom: 1px solid #e5e5ea;
    padding-bottom: 1.25rem;
}
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1c1c1e;
    margin: 0 0 0.2rem 0;
    line-height: 1.2;
}
.page-sub {
    font-size: 0.88rem;
    color: #6e6e73;
    margin: 0;
}

/* ── INPUT AREA ── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #d1d1d6 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    color: #1c1c1e !important;
    line-height: 1.65 !important;
}
.stTextArea textarea:focus { border-color: #636366 !important; }
.stTextInput input {
    background: #ffffff !important;
    border: 1.5px solid #d1d1d6 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    color: #1c1c1e !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #1c1c1e !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 0.55rem 1.75rem !important; letter-spacing: 0.01em !important;
}
.stButton > button:hover { opacity: 0.82 !important; }
.stDownloadButton > button {
    background: #ffffff !important; color: #1c1c1e !important;
    border: 1.5px solid #d1d1d6 !important; border-radius: 8px !important;
    font-weight: 500 !important; font-size: 0.82rem !important;
}
.stDownloadButton > button:hover { border-color: #636366 !important; }

/* ── CARDS ── */
.card {
    background: #ffffff; border: 1px solid #e5e5ea;
    border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
.card-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8e8e93; margin-bottom: 0.85rem;
}

/* ── TOPIC BADGE ── */
.topic-badge {
    display: inline-block; background: #1c1c1e; color: #ffffff;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; padding: 4px 12px;
    border-radius: 4px; margin-bottom: 1.25rem;
}

/* ── WORD COUNT BAR ── */
.wc-row {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: #3a3a3c; margin-bottom: 6px;
}
.wc-bar-bg {
    background: #f2f2f7; border-radius: 99px;
    height: 6px; overflow: hidden; margin-bottom: 4px;
}
.wc-bar-fill { height: 6px; border-radius: 99px; background: #1c1c1e; }
.wc-note { font-size: 0.73rem; color: #8e8e93; text-align: right; }

/* ── KEYWORD TAGS ── */
.kw-wrap { line-height: 2.4; }
.kw-tag {
    display: inline-block; font-size: 0.73rem; font-weight: 500;
    padding: 3px 10px; border-radius: 4px; margin: 2px 3px; border: 1px solid;
}
.kw-a { background: #fff0f0; color: #c0392b; border-color: #f5c6c2; }
.kw-b { background: #f0f4ff; color: #2c55a0; border-color: #c5d0f0; }
.kw-c { background: #f0fff4; color: #1e7e34; border-color: #b8e6c4; }
.kw-d { background: #fef9e7; color: #856404; border-color: #f0d98c; }
.kw-e { background: #f5f0ff; color: #6f42c1; border-color: #d4bff0; }

/* ── SUMMARY OUTPUT ── */
.summary-para { font-size: 0.92rem; line-height: 1.85; color: #1c1c1e; }
.bullet-row {
    display: flex; gap: 12px; padding: 9px 0;
    border-bottom: 1px solid #f2f2f7; align-items: flex-start;
}
.bullet-row:last-child { border-bottom: none; }
.bullet-num {
    min-width: 22px; height: 22px; background: #1c1c1e; color: white;
    border-radius: 4px; font-size: 0.68rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-top: 3px; flex-shrink: 0;
}
.bullet-text { font-size: 0.9rem; line-height: 1.75; color: #1c1c1e; }

/* ── URL WARNING BOX ── */
.url-note {
    background: #fff8f0; border: 1px solid #f0d9b8;
    border-radius: 8px; padding: 0.75rem 1rem;
    font-size: 0.82rem; color: #7a4f00; margin-bottom: 1rem;
}

/* ── FOOTER ── */
.site-footer {
    margin-top: 4rem;
    padding: 1.5rem 0;
    border-top: 1px solid #e5e5ea;
    text-align: center;
    font-size: 0.8rem;
    color: #8e8e93;
}
.site-footer a { color: #636366; text-decoration: none; }
.site-footer a:hover { color: #1c1c1e; }

hr { border-color: #e5e5ea !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ─────────────────────────────────────────────────────
def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords(text, top_n=15):
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    stop.update(['said','also','would','could','one','like','even','well','just',
                 'get','got','make','made','use','upon','though','without','within'])
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop]
    freq = Counter(filtered)
    return [word for word, _ in freq.most_common(top_n)]

def detect_topic(text):
    topics = {
        "Health & Medicine": ["health","medical","doctor","disease","patient","hospital",
                               "treatment","medicine","virus","cancer","drug","symptom","clinical"],
        "Technology": ["technology","software","computer","data","algorithm","artificial",
                       "intelligence","machine","learning","digital","internet","programming","cyber"],
        "Finance & Business": ["financial","market","economy","business","investment","stock",
                                "revenue","profit","bank","fund","trade","economic","company"],
        "Sports": ["sport","game","player","team","match","tournament","league",
                   "score","coach","athlete","championship","win","goal"],
        "Politics": ["government","political","election","president","minister","policy",
                     "law","democracy","parliament","vote","senator","congress"],
        "Science": ["research","study","experiment","scientist","theory","evidence",
                    "discovery","biology","chemistry","physics","journal","findings"],
        "Entertainment": ["movie","film","music","artist","celebrity","entertainment",
                          "award","album","actor","director","show","series"],
        "Environment": ["climate","environment","carbon","emission","energy","sustainable",
                        "pollution","nature","ecosystem","warming","renewable"],
    }
    text_lower = text.lower()
    scores = {t: sum(text_lower.count(kw) for kw in kws) for t, kws in topics.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 2 else "General"

def summarize_text(text, num_sentences, focus=None):
    text = clean_text(text)
    if count_words(text) < 40:
        return [text]

    if focus and focus.strip():
        focus_words = focus.lower().split()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        scored = sorted(sentences, key=lambda s: sum(s.lower().count(w) for w in focus_words), reverse=True)
        text = ' '.join(scored)

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        result = LexRankSummarizer()(parser.document, num_sentences)
        sentences = [str(s) for s in result]
        if sentences:
            return sentences
    except Exception:
        pass

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        result = LsaSummarizer()(parser.document, num_sentences)
        sentences = [str(s) for s in result]
        if sentences:
            return sentences
    except Exception:
        pass

    sents = re.split(r'(?<=[.!?])\s+', text)
    return sents[:num_sentences]

def extract_pdf_text(file):
    reader = PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def fetch_url_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script","style","nav","footer","header","aside","noscript","figure","figcaption"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup
    paragraphs = article.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
    return text

def make_txt(sentences):
    return "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))


# ── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Text Summarizer")
    st.markdown("---")
    mode = st.selectbox("Input Type", ["Text", "PDF Document", "URL"])
    st.markdown("### Settings")
    length_label = st.select_slider(
        "Summary Length",
        options=["Short (3)", "Medium (5)", "Detailed (8)", "Comprehensive (12)"],
        value="Medium (5)"
    )
    length_map = {"Short (3)": 3, "Medium (5)": 5, "Detailed (8)": 8, "Comprehensive (12)": 12}
    num_sentences = length_map[length_label]
    output_mode = st.radio("Output Format", ["Paragraph", "Numbered List"])
    focus_on = st.text_input("Custom Focus", placeholder="e.g. financial impact")
    st.markdown("---")
    st.markdown("<small>Summarize text, PDFs, or web articles. Extracts key phrases and detects topic automatically.</small>",
                unsafe_allow_html=True)


# ── PAGE HEADER ─────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <p class="page-title">Text Summarizer</p>
    <p class="page-sub">Extract key information from any text, document or article</p>
</div>
""", unsafe_allow_html=True)

input_text = ""
do_summarize = False

# ── INPUT ────────────────────────────────────────────────────────
if mode == "Text":
    input_text = st.text_area(
        "Input", height=200,
        placeholder="Paste any article, essay, report or document here...",
        label_visibility="collapsed"
    )
    do_summarize = st.button("Summarize")

elif mode == "PDF Document":
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("Reading PDF..."):
            input_text = extract_pdf_text(uploaded)
        if input_text.strip():
            st.caption(f"{count_words(input_text):,} words extracted")
            do_summarize = st.button("Summarize PDF")
        else:
            st.error("Could not extract text. This PDF may be image-based or scanned.")

elif mode == "URL":
    st.markdown("""<div class="url-note">
        Note: Some websites block automated access (403 error). Try news sites like BBC, Reuters, Wikipedia, or Medium for best results.
    </div>""", unsafe_allow_html=True)
    url = st.text_input("Article URL", placeholder="https://en.wikipedia.org/wiki/...", label_visibility="collapsed")
    if url:
        do_summarize = st.button("Fetch & Summarize")
        if do_summarize:
            with st.spinner("Fetching article..."):
                try:
                    input_text = fetch_url_text(url)
                    if not input_text.strip():
                        st.error("No readable text found on this page. Try pasting the text directly instead.")
                        do_summarize = False
                except requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response else "?"
                    if code == 403:
                        st.error("This website blocks automated access (403). Please copy and paste the article text directly into the Text tab.")
                    elif code == 404:
                        st.error("Page not found (404). Please check the URL.")
                    else:
                        st.error(f"Could not fetch page (Error {code}). Try pasting the text directly instead.")
                    do_summarize = False
                except Exception as e:
                    st.error(f"Could not reach this URL. Try pasting the text directly instead.")
                    do_summarize = False


# ── RESULTS ──────────────────────────────────────────────────────
if do_summarize and input_text and input_text.strip():
    original_wc = count_words(input_text)

    if original_wc < 30:
        st.warning("Text is too short to summarize meaningfully. Please provide at least a paragraph.")
    else:
        with st.spinner("Summarizing..."):
            sentences = summarize_text(input_text, num_sentences, focus_on)
            summary_text = " ".join(sentences)
            summary_wc = count_words(summary_text)
            reduction = round((1 - summary_wc / max(original_wc, 1)) * 100)
            keywords = extract_keywords(input_text)
            topic = detect_topic(input_text)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="topic-badge">{topic}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            # Word count
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Word Count</div>', unsafe_allow_html=True)
            bar_pct = max(4, 100 - reduction)
            st.markdown(f"""
            <div class="wc-row">
                <span>Original: <strong>{original_wc:,} words</strong></span>
                <span>Summary: <strong>{summary_wc:,} words</strong></span>
            </div>
            <div class="wc-bar-bg"><div class="wc-bar-fill" style="width:{bar_pct}%"></div></div>
            <div class="wc-note">{reduction}% reduction</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Keywords
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Key Phrases</div>', unsafe_allow_html=True)
            colors = ["kw-a","kw-b","kw-c","kw-d","kw-e"]
            tags = "".join(f'<span class="kw-tag {colors[i%5]}">{kw}</span>' for i, kw in enumerate(keywords))
            st.markdown(f'<div class="kw-wrap">{tags}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("View original text"):
                st.write(input_text[:4000] + ("..." if len(input_text) > 4000 else ""))

        with col2:
            # Summary output
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Summary</div>', unsafe_allow_html=True)
            if output_mode == "Numbered List":
                html = "".join(
                    f'<div class="bullet-row"><div class="bullet-num">{i+1}</div><div class="bullet-text">{s}</div></div>'
                    for i, s in enumerate(sentences)
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="summary-para">{summary_text}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Download .txt", data=make_txt(sentences),
                    file_name="summary.txt", mime="text/plain",
                    use_container_width=True
                )
            with b2:
                safe = summary_text.replace('`','').replace('\n',' ')
                st.markdown(f"""<button onclick="navigator.clipboard.writeText(`{safe}`).then(()=>{{this.textContent='Copied';setTimeout(()=>this.textContent='Copy to Clipboard',2000)}})"
                style="width:100%;padding:0.43rem 0.5rem;background:#fff;color:#1c1c1e;border:1.5px solid #d1d1d6;border-radius:8px;font-size:0.82rem;font-weight:500;cursor:pointer;"
                onmouseover="this.style.borderColor='#636366'" onmouseout="this.style.borderColor='#d1d1d6'">Copy to Clipboard</button>""",
                unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    Made with ♥ by <strong>Isha Ghag</strong> &nbsp;|&nbsp;
</div>
""", unsafe_allow_html=True)
