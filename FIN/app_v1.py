# app_v1.py 
import json
import streamlit as st
import finance_agent as fa  # your existing logic

st.set_page_config(
    page_title="Liquidity & Solvency Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",  # keep hidden; users can open with toggle
)

# Dark mode friendly 
st.markdown("""
<style>
html, body, [class*="stApp"] { background-color:#0f1117; color:#e5e7eb; }
h1,h2,h3,h4,h5,h6 { color:#e5e7eb; }
.block-container { padding-top:2rem; }
/* center text inside the input field */
div[data-baseweb="input"] input { text-align:center; }
/* remove extra top spacing before input on first screen */
.center-wrap { max-width: 560px; margin: 0 auto; }
.band-badge { padding:4px 10px; border-radius:9999px; font-size:12px; color:#fff; }
</style>
""", unsafe_allow_html=True)

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

# session
if "started" not in st.session_state:
    st.session_state.started = False
if "ticker" not in st.session_state:
    st.session_state.ticker = ""   # <-- no default "AAPL"

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Open this sidebar to tweak inputs / debug.")
    sb_ticker = st.text_input("Ticker", value=st.session_state.ticker, placeholder="AAPL / 005930.KS / 7203.T")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        reanalyze = st.button("Re-Analyㄴe", use_container_width=True)
    with col_sb2:
        show_json = st.toggle("Show JSON", value=False)
    # small help
    st.caption("💡 KR/JP/HK 종목은 .KS / .T / .HK 접미사를 붙여주세요.")
    st.caption("💡 Powered by 'Finance_Agent'")

# If user triggers from sidebar
if 'show_json' not in st.session_state:
    st.session_state.show_json = False
if 'sb_ticker' not in st.session_state:
    st.session_state.sb_ticker = st.session_state.ticker

if reanalyze:
    st.session_state.started = True
    st.session_state.ticker = (sb_ticker or "").strip().upper()
    st.session_state.show_json = show_json
    st.rerun()

# First Screen 
if not st.session_state.started:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.title("📊 Liquidity & Solvency Analyser")
    st.caption("Enter a ticker symbol to analyse liquidity & solvency ratios.")
    st.caption("Powered by 'Finance_Agent'")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
    ticker = st.text_input(
        "Ticker",
        value="",                      # <-- empty by default
        placeholder="e.g., AAPL, MSFT, 005930.KS, 7203.T",
        key="center_ticker",
        label_visibility="collapsed"   # <-- remove the label bar (no grey header)
    )
    go = st.button("🔎 Analyze", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not go:
        st.stop()

    st.session_state.ticker = (ticker or "").strip().upper()
    st.session_state.started = True
    st.rerun()

# Results Screen
st.title("📊 Liquidity & Solvency Analyser")

query = build_query(st.session_state.ticker)
if hasattr(fa, "run_query"):
    result = fa.run_query(query)
else:
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

core   = result.get("core", {})
ratios = core.get("ratios", {})
liq    = ratios.get("Liquidity", {})
sol    = ratios.get("Solvency", {})

# Summary
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

st.markdown("### 💧 Liquidity")
cL1, cL2, cL3 = st.columns(3)
with cL1: ratio_card("Current Ratio", liq.get("current_ratio",{}))
with cL2: ratio_card("Quick Ratio",   liq.get("quick_ratio",{}))
with cL3: ratio_card("Cash Ratio",    liq.get("cash_ratio",{}))

st.markdown("### 🛡️ Solvency")
cS1, cS2, cS3 = st.columns(3)
with cS1: ratio_card("Debt-to-Equity",    sol.get("debt_to_equity",{}))
with cS2: ratio_card("Debt Ratio",        sol.get("debt_ratio",{}))
with cS3: ratio_card("Interest Coverage", sol.get("interest_coverage",{}))

st.markdown("### 📝 Narrative")
st.write(result.get("explanation", "—"))
