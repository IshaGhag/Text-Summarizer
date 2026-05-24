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

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background: #f9f7f4;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #1c1c1e !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #e5e5e7 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
/* Fix dropdown text visibility */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: #2c2c2e !important;
    border-color: #3a3a3c !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
}
/* Fix text input visibility */
[data-testid="stSidebar"] .stTextInput input {
    background: #2c2c2e !important;
    border: 1px solid #3a3a3c !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #8e8e93 !important;
}
/* Slider */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    margin-top: 8px;
}
/* Radio */
[data-testid="stSidebar"] .stRadio label {
    color: #e5e5e7 !important;
}
/* Divider */
[data-testid="stSidebar"] hr {
    border-color: #3a3a3c !important;
}

/* ── MAIN AREA ── */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px;
}

/* Page title */
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1c1c1e;
    margin-bottom: 0.1rem;
}
.page-sub {
    font-size: 0.9rem;
    color: #6e6e73;
    margin-bottom: 1.5rem;
}

/* Text area */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #d1d1d6 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    color: #1c1c1e !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #636366 !important;
    box-shadow: 0 0 0 3px rgba(99,99,102,0.1) !important;
}

/* Summarize button */
.stButton > button {
    background: #1c1c1e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.75rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.82 !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: #ffffff !important;
    color: #1c1c1e !important;
    border: 1.5px solid #d1d1d6 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 1rem !important;
}
.stDownloadButton > button:hover {
    border-color: #636366 !important;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8e8e93;
    margin-bottom: 0.9rem;
}

/* Topic badge */
.topic-badge {
    display: inline-block;
    background: #1c1c1e;
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 4px;
    margin-bottom: 1.25rem;
}

/* Word count bar */
.wc-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #3a3a3c;
    margin-bottom: 6px;
}
.wc-bar-bg {
    background: #f2f2f7;
    border-radius: 99px;
    height: 6px;
    margin-bottom: 4px;
    overflow: hidden;
}
.wc-bar-fill {
    height: 6px;
    border-radius: 99px;
    background: #1c1c1e;
}
.wc-reduction {
    font-size: 0.75rem;
    color: #636366;
    text-align: right;
}

/* Keyword tags */
.kw-wrap { margin-top: 0.5rem; line-height: 2.2; }
.kw-tag {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 4px;
    margin: 2px 3px;
    border: 1px solid;
}
.kw-a { background: #fff0f0; color: #c0392b; border-color: #f5c6c2; }
.kw-b { background: #f0f4ff; color: #2c55a0; border-color: #c5d0f0; }
.kw-c { background: #f0fff4; color: #1e7e34; border-color: #b8e6c4; }
.kw-d { background: #fef9e7; color: #856404; border-color: #f0d98c; }
.kw-e { background: #f5f0ff; color: #6f42c1; border-color: #d4bff0; }

/* Summary text */
.summary-para {
    font-size: 0.93rem;
    line-height: 1.85;
    color: #1c1c1e;
}

/* Bullet list */
.bullet-row {
    display: flex;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid #f2f2f7;
    align-items: flex-start;
}
.bullet-row:last-child { border-bottom: none; }
.bullet-num {
    min-width: 22px; height: 22px;
    background: #1c1c1e; color: white;
    border-radius: 4px; font-size: 0.7rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-top: 3px; flex-shrink: 0;
}
.bullet-text {
    font-size: 0.9rem; line-height: 1.75; color: #1c1c1e;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1.5px dashed #d1d1d6 !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

/* Divider */
hr { border-color: #e5e5ea !important; }

/* Success / info boxes */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ─────────────────────────────────────────────────────

def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    return text.strip()

def extract_keywords(text, top_n=15):
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    stop.update(['said','also','would','could','one','like','even','well','just','get','got','make','made','use'])
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop]
    freq = Counter(filtered)
    return [word for word, _ in freq.most_common(top_n)]

def detect_topic(text):
    topics = {
        "Health & Medicine": ["health","medical","doctor","disease","patient","hospital","treatment","medicine","virus","cancer","drug","symptom","clinical","therapy","diagnosis"],
        "Technology": ["technology","software","computer","data","algorithm","artificial","intelligence","machine","learning","digital","internet","code","programming","cyber","platform","system","model"],
        "Finance & Business": ["financial","market","economy","business","investment","stock","revenue","profit","bank","fund","trade","economic","price","company","growth","earnings"],
        "Sports": ["sport","game","player","team","match","tournament","league","score","coach","athlete","championship","win","goal","season","club"],
        "Politics": ["government","political","election","president","minister","policy","law","democracy","parliament","vote","senator","congress","party","official"],
        "Science": ["research","study","experiment","scientist","theory","evidence","discovery","biology","chemistry","physics","journal","findings","published"],
        "Entertainment": ["movie","film","music","artist","celebrity","entertainment","award","album","actor","director","show","series","streaming"],
        "Environment": ["climate","environment","carbon","emission","energy","sustainable","pollution","nature","ecosystem","warming","fossil","renewable"],
    }
    text_lower = text.lower()
    scores = {t: sum(text_lower.count(kw) for kw in kws) for t, kws in topics.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 2 else "General"

def summarize_text(text, num_sentences, focus=None):
    """Use LexRank for better quality summaries, with focus boosting."""
    text = clean_text(text)

    # If text is too short, just return it
    words = count_words(text)
    if words < 50:
        return [text]

    # Focus boosting — reorder sentences to put focus-relevant ones first
    if focus and focus.strip():
        focus_words = focus.lower().split()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        scored = []
        for s in sentences:
            score = sum(s.lower().count(w) for w in focus_words)
            scored.append((score, s))
        scored.sort(reverse=True)
        # Rebuild text with boosted sentences first
        text = ' '.join(s for _, s in scored)

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        # Try LexRank first (better quality)
        summarizer = LexRankSummarizer()
        result = summarizer(parser.document, num_sentences)
        sentences = [str(s) for s in result]
        if sentences:
            return sentences
    except Exception:
        pass

    # Fallback to LSA
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        result = summarizer(parser.document, num_sentences)
        return [str(s) for s in result]
    except Exception:
        # Last resort: return first N sentences
        sents = re.split(r'(?<=[.!?])\s+', text)
        return sents[:num_sentences]

def extract_pdf_text(file):
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)
    return " ".join(pages_text)

def fetch_url_text(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove nav, footer, scripts
    for tag in soup(["script","style","nav","footer","header","aside","noscript"]):
        tag.decompose()
    # Get article/main content first, fallback to all paragraphs
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
    st.markdown("<small style='color:#636366'>Summarize text, PDFs, or web articles. Extracts key phrases and detects topic automatically.</small>", unsafe_allow_html=True)


# ── HEADER ──────────────────────────────────────────────────────
st.markdown('<p class="page-title">Text Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Extract key information from any text, document or article</p>', unsafe_allow_html=True)

input_text = ""
do_summarize = False

# ── INPUT AREA ──────────────────────────────────────────────────
if mode == "Text":
    input_text = st.text_area(
        "Input text",
        height=200,
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
            st.caption(f"{count_words(input_text):,} words extracted from PDF")
            do_summarize = st.button("Summarize PDF")
        else:
            st.error("Could not extract text from this PDF. It may be scanned or image-based.")

elif mode == "URL":
    url = st.text_input("Article URL", placeholder="https://...", label_visibility="collapsed")
    if url:
        do_summarize = st.button("Fetch & Summarize")
        if do_summarize:
            with st.spinner("Fetching article..."):
                try:
                    input_text = fetch_url_text(url)
                    if not input_text.strip():
                        st.error("Could not extract content from this URL.")
                        do_summarize = False
                except Exception as e:
                    st.error(f"Failed to fetch URL: {str(e)}")
                    do_summarize = False


# ── RESULTS ─────────────────────────────────────────────────────
if do_summarize and input_text and input_text.strip():
    original_wc = count_words(input_text)

    if original_wc < 30:
        st.warning("Text is too short to summarize. Please provide at least a few paragraphs.")
    else:
        with st.spinner("Summarizing..."):
            sentences = summarize_text(input_text, num_sentences, focus_on)
            summary_text = " ".join(sentences)
            summary_wc = count_words(summary_text)
            reduction = round((1 - summary_wc / max(original_wc, 1)) * 100)
            keywords = extract_keywords(input_text)
            topic = detect_topic(input_text)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Topic badge
        st.markdown(f'<div class="topic-badge">{topic}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        # ── LEFT: Stats & Keywords ──
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Word Count</div>', unsafe_allow_html=True)
            bar_pct = max(4, 100 - reduction)
            st.markdown(f"""
            <div class="wc-row">
                <span>Original: <strong>{original_wc:,} words</strong></span>
                <span>Summary: <strong>{summary_wc:,} words</strong></span>
            </div>
            <div class="wc-bar-bg">
                <div class="wc-bar-fill" style="width:{bar_pct}%"></div>
            </div>
            <div class="wc-reduction">{reduction}% reduction</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Key Phrases</div>', unsafe_allow_html=True)
            colors = ["kw-a","kw-b","kw-c","kw-d","kw-e"]
            tags = "".join(
                f'<span class="kw-tag {colors[i % len(colors)]}">{kw}</span>'
                for i, kw in enumerate(keywords)
            )
            st.markdown(f'<div class="kw-wrap">{tags}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("View original text"):
                st.write(input_text[:4000] + ("..." if len(input_text) > 4000 else ""))

        # ── RIGHT: Summary ──
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Summary</div>', unsafe_allow_html=True)

            if output_mode == "Numbered List":
                html = ""
                for i, s in enumerate(sentences):
                    html += f'<div class="bullet-row"><div class="bullet-num">{i+1}</div><div class="bullet-text">{s}</div></div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="summary-para">{summary_text}</p>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Action buttons
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Download .txt",
                    data=make_txt(sentences),
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with b2:
                copy_js = f"""<button onclick="navigator.clipboard.writeText(`{summary_text.replace('`','').replace(chr(10),' ')}`).then(()=>{{this.textContent='Copied';setTimeout(()=>this.textContent='Copy to Clipboard',2000)}})" style="width:100%;padding:0.45rem 0.5rem;background:#fff;color:#1c1c1e;border:1.5px solid #d1d1d6;border-radius:8px;font-size:0.82rem;font-weight:500;cursor:pointer;transition:border-color 0.15s;" onmouseover="this.style.borderColor='#636366'" onmouseout="this.style.borderColor='#d1d1d6'">Copy to Clipboard</button>"""
                st.markdown(copy_js, unsafe_allow_html=True)
