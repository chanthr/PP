# app_v1.py
import os
import json
import streamlit as st
import finance_agent as fa  # <-- 방금 만든 LangChain 기반 agent

# -------------------------
# Page config + styles
# -------------------------
st.set_page_config(
    page_title="Liquidity & Solvency Analysis",
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
    st.markdown("### ⚙️ Controls")
    st.caption("Open this sidebar to tweak inputs / debug.")

    # Language selector
    lang_label = st.selectbox("Language", ["한국어", "English"], index=0, help="Narrative language")
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
        💡 KR/JP/HK 종목은 <code>.KS</code> / <code>.T</code> / <code>.HK</code> 접미사를 붙이세요.<br><br>
        💡 <i>Powered by Finance_Agent (LangChain + Groq)</i>
        </div>
        """,
        unsafe_allow_html=True
    )

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
    st.title("📊 Liquidity & Solvency Analyser")
    st.caption("Enter a ticker symbol to analyse liquidity & solvency ratios.")
    st.caption("Powered by Finance_Agent")
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
st.title("📊 Liquidity & Solvency Analyser")

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
st.write(result.get("explanation", "—"))

# Optional: raw JSON view
if st.session_state.show_json:
    st.markdown("### 🔍 Raw JSON")
    st.json(result)
