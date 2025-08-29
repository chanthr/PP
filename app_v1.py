# app_v1.py
import re
import os
import json
import streamlit as st
import finance_agent as fa  


### --- Update log --- ###

CHANGELOG_MD = """
### ✅ v0.1.0 — LLM Integration & Fallback Improvements
- Store ChatGroq initialization failure reasons internally and display in UI (easier to identify fallback cause)
- Restructure to allow immediate retry after replacing GROQ_API_KEY at runtime
- Cleaned up Narrative section formatting/layout

### ✅ v0.0.5 — Fallback Summary Improvements
- Company description limited to 30 words (prevents long output during fallback)
- Overall financial health rating logic (Excellent / Good / Average / Weak)

### ✅ v0.0.1 — Initial Release
- Liquidity/Solvency ratio calculations
- Narrative generation (LLM/Fallback)
- Basic Streamlit UI
"""

############################


# -------------------------
# Page config + styles
# -------------------------
st.set_page_config(
    page_title="LSA Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="stApp"] { background-color:#0f1117; color:#e5e7eb; }
h1,h2,h3,h4,h5,h6 { color:#e5e7eb; }
.block-container { padding-top:2rem; }
div[data-baseweb="input"] input { text-align:center; }
.center-wrap { max-width: 560px; margin: 0 auto; }
.band-badge { padding:4px 10px; border-radius:9999px; font-size:12px; color:#fff; }
.small-muted { font-size: 13px; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
def fmt_ratio(x):
    if x is None: return "N/A"
    try: return f"{float(x):.2f}"
    except Exception: return "N/A"

def badge(band):
    color = {"Strong":"#10b981","Fair":"#f59e0b","Weak":"#ef4444","N/A":"#6b7280"}.get(band,"#6b7280")
    st.markdown(f'<span class="band-badge" style="background:{color}">{band}</span>',
                unsafe_allow_html=True)

def ratio_card(title, node):
    node = node or {}
    cols = st.columns([2,1])
    cols[0].markdown(f"**{title}**")
    with cols[1]: badge(node.get("band","N/A"))
    st.metric("Value", fmt_ratio(node.get("value")))

def build_query(t: str) -> str:
    return f"{(t or '').strip()} 유동성/건전성 평가"

# -------------------------
# Session init
# -------------------------
if "started" not in st.session_state:
    st.session_state.started = False
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "show_json" not in st.session_state:
    st.session_state.show_json = False

# -------------------------
# Sidebar (controls)
# -------------------------
with st.sidebar:
    status = getattr(fa, "llm", None)
    if status is None:
        st.info("LLM: Fallback mode (no key or init failure).")
    else:
        st.success("LLM: Connected (Groq).")

    # Header
    st.markdown("### ⚙️ Controls")

    # Page Selector 
    page = st.radio("", ["📈 Analysis", "📝 Updates"], index=0)
    if page == "📝 Updates":
        st.title("📝 Patch Notes")   # ← 이거 하나만 남김
        st.markdown(CHANGELOG_MD, unsafe_allow_html=True)
        st.stop()  # 다른 화면 로직 실행 방지
    
    # Language selector
    lang_label = st.selectbox("Language", ["Korean", "English"], index=0, help="Narrative language")
    lang_code = "ko" if lang_label == "한국어" else "en"

    # Ticker input
    sb_ticker = st.text_input(
        "Ticker",
        value=st.session_state.ticker,
        placeholder="AAPL / 005930.KS / 7203.T",
        key="sb_ticker",
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        reanalyze = st.button("🔄 Re-Analyse", use_container_width=True)
    with c2:
        st.session_state.show_json = st.toggle("Show JSON", value=st.session_state.show_json)

    st.markdown(
        """
        <div class="small-muted" style="margin-top: 1rem;">
        💡 KR/JP/HK 종목은 <code>.KS</code> / <code>.T</code> / <code>.HK</code> 접미사를 붙여주세요.<br><br>
        💡 <i>Powered by finance agent v1</i>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.caption("Use your Groq API Key")
    new_key = st.text_input("", type="password")
    if st.button("Use this key"):
        if new_key.strip():
            fa.set_runtime_groq_key(new_key.strip())  # 🔧 런타임 주입
            st.success("GROQ key set. Re-run analysis.")
        else:
            st.warning("Please paste a non-empty key.")
    
# Sidebar action
if reanalyze:
    st.session_state.started = True
    st.session_state.ticker = (sb_ticker or "").strip().upper()
    st.rerun()

# -------------------------
# First screen (center input)
# -------------------------
if not st.session_state.started:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.title("📊 LSA Agent")
    st.caption("Enter a ticker symbol to analyse liquidity & solvency ratios.")
    st.caption("Powered by finance agent v1")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
    ticker = st.text_input(
        "Ticker",
        value="",
        placeholder="e.g., AAPL, MSFT, 005930.KS, 7203.T",
        key="center_ticker",
        label_visibility="collapsed"
    )
    go = st.button("🔎 Analyse", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not go:
        st.stop()

    st.session_state.ticker = (ticker or "").strip().upper()
    if not st.session_state.ticker:
        st.warning("⚠️ Please enter a ticker (e.g., AAPL, 005930.KS).")
        st.stop()
    st.session_state.started = True
    st.rerun()

# -------------------------
# Results screen
# -------------------------
st.title("📊 LSA Agent")

query = build_query(st.session_state.ticker)

# 분석 실행
try:
    with st.spinner("Analyzing ratios & generating narrative..."):
        if hasattr(fa, "run_query"):
            result = fa.run_query(query, language=("ko" if lang_label == "한국어" else "en"))
        else:
            # 아주 예전 버전 대비 폴백
            payload = fa.compute_ratios_for_ticker(st.session_state.ticker)
            result = {
                "core": {
                    "company": payload.get("company"),
                    "ticker": payload.get("ticker"),
                    "price": payload.get("price"),
                    "ratios": payload.get("ratios", {}),
                },
                "notes": payload.get("notes"),
                "explanation": "",
            }
except Exception as e:
    st.error(f"❌ Failed to analyse: {e}")
    st.stop()

core   = result.get("core", {}) or {}
ratios = core.get("ratios", {}) or {}
liq    = ratios.get("Liquidity", {}) or {}
sol    = ratios.get("Solvency", {}) or {}

# Summary row
cols = st.columns([3,1,1])
with cols[0]:
    st.subheader(f"{core.get('company','-')}  ({core.get('ticker','-')})")
    if result.get("notes"):
        st.info(result["notes"])
with cols[1]:
    price = core.get("price")
    st.metric("Last Price", "N/A" if price is None else f"{price:,.2f}")
with cols[2]:
    cr = liq.get("current_ratio", {})
    st.metric("Liquidity Score", fmt_ratio(cr.get("value")), cr.get("band","N/A"))

# Liquidity
st.markdown("### 💧 Liquidity")
cL1, cL2, cL3 = st.columns(3)
with cL1: ratio_card("Current Ratio", liq.get("current_ratio",{}))
with cL2: ratio_card("Quick Ratio",   liq.get("quick_ratio",{}))
with cL3: ratio_card("Cash Ratio",    liq.get("cash_ratio",{}))

# Solvency
st.markdown("### 🛡️ Solvency")
cS1, cS2, cS3 = st.columns(3)
with cS1: ratio_card("Debt-to-Equity",    sol.get("debt_to_equity",{}))
with cS2: ratio_card("Debt Ratio",        sol.get("debt_ratio",{}))
with cS3: ratio_card("Interest Coverage", sol.get("interest_coverage",{}))

# Narrative
st.markdown("### 📝 Narrative")

def _prettify_markdown(narr: str) -> str:
    s = (narr or "—").strip()

    # 1) 중간에 박혀있는 " • " 불릿을 마크다운 리스트로 변환
    s = s.replace(" • ", "\n\n- ")
    s = s.replace("\n• ", "\n\n- ")
    s = s.replace("• ", "- ")  # 문장 시작 불릿 보정

    # 2) 주요 섹션 라벨을 굵게 + 줄바꿈
    s = re.sub(r"\b(회사 개요:)", r"**\1**\n\n", s)
    s = re.sub(r"\b(유동성:)",   r"\n\n**\1** ", s)
    s = re.sub(r"\b(건전성:)",   r"\n\n**\1** ", s)
    s = re.sub(r"\b(한줄평:)",   r"\n\n**\1** ", s)

    # 3) 너무 긴 한 덩어리일 때. 마침표 뒤에 두 칸 개행을 넣어 문단 분리(과도하면 주석처리)
    # s = re.sub(r"\. ", ".\n\n", s)

    return s

narr = result.get("explanation", "—") or "—"
st.markdown(_prettify_markdown(narr))

# Optional: raw JSON view
if st.session_state.show_json:
    st.markdown("### 🔍 Raw JSON")
    st.json(result)

# app_v1.py 결과 아래에 표시 (요약 행 근처)
src = (result.get("meta") or {}).get("source")
if src == "fallback":
    st.caption("⚙️ Narrative generated with fallback (LLM unavailable).")
