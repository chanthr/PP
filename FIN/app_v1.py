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
/* 기본 배경/글자색: 라이트 모드 */
html, body, [class*="stApp"] { 
    background-color: #ffffff; 
    color: #111827;   /* 거의 검은색 */
}

/* 제목 계열 */
h1,h2,h3,h4,h5,h6 { 
    color: #1f2937;   /* 진회색 → 가독성 좋음 */
}

/* 컨테이너 간격 */
.block-container { 
    padding-top: 2rem; 
}

/* 입력창 텍스트 중앙정렬 */
div[data-baseweb="input"] input { 
    text-align: center; 
    color: #111827; 
    background-color: #f9fafb;   /* 아주 연한 회색 배경 */
    border: 1px solid #d1d5db;   /* 연회색 테두리 */
    border-radius: 6px;
}

/* 첫 화면 입력 필드 정렬 */
.center-wrap { 
    max-width: 560px; 
    margin: 0 auto; 
}

/* Band badge (Strong/Fair/Weak) 색상 유지 */
.band-badge { 
    padding: 4px 10px; 
    border-radius: 9999px; 
    font-size: 12px; 
    color: #ffffff; 
}
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
