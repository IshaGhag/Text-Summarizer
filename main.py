import streamlit as st
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import re
from collections import Counter
import io

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #fdf6ec 0%, #fff8f5 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1035 0%, #2d1f0e 100%) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #f0ede8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label { color: #c8b89a !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important; border-radius: 8px !important;
}

/* Cards */
.summary-card {
    background: white; border-radius: 16px; padding: 1.5rem;
    border: 1.5px solid rgba(240,165,0,0.15);
    box-shadow: 0 4px 20px rgba(200,120,60,0.08);
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(135deg, #fff8f0, #fff3e8);
    border-radius: 16px; padding: 1.5rem;
    border: 1.5px solid rgba(240,165,0,0.2);
    box-shadow: 0 4px 20px rgba(200,120,60,0.1);
}

/* Keyword tags */
.keyword-tag {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 3px;
    cursor: default;
}
.tag-0 { background: #fde8d8; color: #c05c00; }
.tag-1 { background: #e8f4fd; color: #0066aa; }
.tag-2 { background: #e8fde8; color: #006600; }
.tag-3 { background: #fde8fd; color: #880088; }
.tag-4 { background: #fdfde8; color: #888800; }

/* Topic badge */
.topic-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #e8956d, #f0a500);
    color: white; padding: 6px 16px; border-radius: 99px;
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.03em;
    margin-bottom: 1rem;
}

/* Word count bar */
.wc-bar-wrap {
    background: #f5ece0; border-radius: 99px; height: 10px;
    margin: 8px 0; overflow: hidden;
}
.wc-bar-fill {
    height: 10px; border-radius: 99px;
    background: linear-gradient(90deg, #e8956d, #f0a500);
    transition: width 0.6s ease;
}

/* Bullet list */
.bullet-item {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px solid rgba(240,165,0,0.1);
}
.bullet-num {
    background: linear-gradient(135deg, #e8956d, #f0a500);
    color: white; border-radius: 50%; width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e8956d, #f0a500) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Download buttons */
.stDownloadButton > button {
    background: white !important; color: #e8956d !important;
    border: 1.5px solid #e8956d !important; border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Title */
.main-title {
    font-size: 2.2rem; font-weight: 700; color: #2d1f0e;
    margin-bottom: 0.2rem;
}
.main-sub { color: #8a6a50; font-size: 1rem; margin-bottom: 1.5rem; }

/* Text areas */
.stTextArea textarea {
    border-radius: 12px !important; border: 1.5px solid rgba(240,165,0,0.25) !important;
    background: white !important;
}
.stTextArea textarea:focus { border-color: #e8956d !important; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────

def count_words(text):
    return len(text.split())

def extract_keywords(text, top_n=12):
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop]
    freq = Counter(filtered)
    return [word for word, _ in freq.most_common(top_n)]

def detect_topic(text):
    topics = {
        "🏥 Health & Medicine": ["health","medical","doctor","disease","patient","hospital","treatment","medicine","virus","cancer","drug","symptom","clinical"],
        "💻 Technology": ["technology","software","computer","data","algorithm","ai","machine","learning","digital","internet","code","programming","tech","cyber"],
        "💰 Finance & Business": ["financial","market","economy","business","investment","stock","revenue","profit","bank","fund","trade","economic","price"],
        "⚽ Sports": ["sport","game","player","team","match","tournament","league","score","coach","athlete","championship","win","goal"],
        "🌍 Politics": ["government","political","election","president","minister","policy","law","democracy","parliament","vote","senator","congress"],
        "🔬 Science": ["research","study","experiment","scientist","theory","evidence","discovery","biology","chemistry","physics","journal"],
        "🎬 Entertainment": ["movie","film","music","artist","celebrity","entertainment","award","album","actor","director","show","series"],
        "🌱 Environment": ["climate","environment","carbon","emission","energy","sustainable","pollution","nature","biodiversity","ecosystem"],
    }
    text_lower = text.lower()
    scores = {topic: sum(text_lower.count(kw) for kw in kws) for topic, kws in topics.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "📝 General"

def summarize(text, num_sentences, focus=None):
    if focus and focus.strip():
        focus_words = focus.lower().split()
        sentences = text.split('. ')
        boosted = []
        for s in sentences:
            score = sum(s.lower().count(w) for w in focus_words)
            boosted.append((score, s))
        boosted.sort(reverse=True)
        top = [s for _, s in boosted[:max(3, len(sentences)//2)]]
        text = '. '.join(top) + ('. ' + '. '.join(s for _, s in boosted[max(3, len(sentences)//2):]))
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return [str(s) for s in summary]

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

def fetch_text_from_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join([p.get_text() for p in soup.find_all("p")])

def make_txt(summary_sentences):
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(summary_sentences))

def make_pdf_bytes(summary_sentences, title="Summary"):
    lines = [title, "=" * len(title), ""]
    for i, s in enumerate(summary_sentences):
        lines.append(f"{i+1}. {s}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Text Summarizer")
    st.markdown("---")

    mode = st.selectbox("📥 Input Type", ["Summarize Text", "Summarize Document"])

    st.markdown("### ⚙️ Summary Settings")

    length_label = st.select_slider(
        "Summary Length",
        options=["Short (3 sentences)", "Medium (5 sentences)", "Detailed (8 sentences)", "Comprehensive (12 sentences)"],
        value="Medium (5 sentences)"
    )
    length_map = {
        "Short (3 sentences)": 3,
        "Medium (5 sentences)": 5,
        "Detailed (8 sentences)": 8,
        "Comprehensive (12 sentences)": 12
    }
    num_sentences = length_map[length_label]

    output_mode = st.radio("📋 Output Format", ["Paragraph", "Bullet Points"])

    focus_on = st.text_input("🎯 Custom Focus (optional)", placeholder="e.g. financial impact")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("<small>Paste text, upload a PDF, or enter a URL. Get instant intelligent summaries with keywords and topic detection.</small>", unsafe_allow_html=True)


# ─── MAIN CONTENT ──────────────────────────────────────────
st.markdown('<p class="main-title">📄 Text Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Intelligent summaries with keyword extraction, topic detection & more</p>', unsafe_allow_html=True)

input_text = ""

if mode == "Summarize Text":
    input_text = st.text_area("Paste your text here", height=220, placeholder="Paste any article, paragraph, or document here...")
    do_summarize = st.button("✨ Summarize", use_container_width=False)

else:
    upload_option = st.radio("Choose input method", ("Upload PDF", "Input URL"), horizontal=True)
    do_summarize = False

    if upload_option == "Upload PDF":
        uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded:
            input_text = extract_text_from_pdf(uploaded)
            st.success(f"✅ PDF loaded — {count_words(input_text):,} words extracted")
            do_summarize = st.button("✨ Summarize PDF", use_container_width=False)
    else:
        url = st.text_input("Enter article URL", placeholder="https://...")
        if url:
            if st.button("✨ Fetch & Summarize"):
                with st.spinner("Fetching page..."):
                    try:
                        input_text = fetch_text_from_url(url)
                        do_summarize = True
                    except Exception as e:
                        st.error(f"Could not fetch URL: {e}")


# ─── RESULTS ───────────────────────────────────────────────
if do_summarize and input_text and input_text.strip():
    original_wc = count_words(input_text)

    with st.spinner("Summarizing..."):
        summary_sentences = summarize(input_text, num_sentences, focus_on)
        summary_text = " ".join(summary_sentences)
        summary_wc = count_words(summary_text)
        reduction = round((1 - summary_wc / original_wc) * 100) if original_wc > 0 else 0
        keywords = extract_keywords(input_text)
        topic = detect_topic(input_text)

    st.markdown("---")

    # ── Topic badge ──
    st.markdown(f'<div class="topic-badge">{topic}</div>', unsafe_allow_html=True)

    # ── Two column layout ──
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📊 Stats & Keywords")
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)

        # Word count comparison
        st.markdown(f"**Word Count**")
        st.markdown(f"Original: **{original_wc:,} words** → Summary: **{summary_wc:,} words** ({reduction}% reduction)")
        pct = max(5, 100 - reduction)
        st.markdown(f"""
        <div class="wc-bar-wrap">
            <div class="wc-bar-fill" style="width:{pct}%"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Keywords
        st.markdown("**🔑 Key Phrases**")
        colors = ["tag-0","tag-1","tag-2","tag-3","tag-4"]
        tags_html = "".join(
            f'<span class="keyword-tag {colors[i % len(colors)]}">{kw}</span>'
            for i, kw in enumerate(keywords)
        )
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Original text preview
        with st.expander("📖 View Original Text"):
            st.write(input_text[:3000] + ("..." if len(input_text) > 3000 else ""))

    with col2:
        st.markdown("#### ✨ Summary")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if output_mode == "Bullet Points":
            bullets_html = ""
            for i, sentence in enumerate(summary_sentences):
                bullets_html += f"""
                <div class="bullet-item">
                    <div class="bullet-num">{i+1}</div>
                    <div style="font-size:0.92rem;line-height:1.7;color:#2d1f0e">{sentence}</div>
                </div>"""
            st.markdown(bullets_html, unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="font-size:0.95rem;line-height:1.85;color:#2d1f0e">{summary_text}</p>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        btn1, btn2, btn3 = st.columns(3)

        with btn1:
            st.download_button(
                "📥 Download .txt",
                data=make_txt(summary_sentences),
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        with btn2:
            st.download_button(
                "📄 Download .pdf",
                data=make_pdf_bytes(summary_sentences),
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        with btn3:
            # Copy to clipboard via JS
            copy_js = f"""
            <button onclick="navigator.clipboard.writeText(`{summary_text.replace('`','')}`).then(()=>{{this.textContent='✅ Copied!';setTimeout(()=>this.textContent='📋 Copy',2000)}})"
            style="width:100%;padding:0.45rem;background:white;color:#e8956d;border:1.5px solid #e8956d;border-radius:10px;font-weight:600;font-size:0.85rem;cursor:pointer;">
            📋 Copy
            </button>"""
            st.markdown(copy_js, unsafe_allow_html=True)

elif do_summarize and not input_text.strip():
    st.warning("Please enter some text first!")
