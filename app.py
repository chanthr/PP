# UI for Finance Agent 
import json
import streamlit as st

# Existing module 
import finance_agent as fa

st.set_page_config(page_title="Liquidity & Solvency Analyzer", page_icon="📊", layout="wide")

# Utility functions 
def fmt_ratio(x):
    if x is None: 
        return "N/A"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"

def badge(label, band):
    color = {"Strong":"green","Fair":"orange","Weak":"red","N/A":"gray"}.get(band, "gray")
    st.markdown(f"<span style='padding:4px 8px;border-radius:12px;background:{color};color:white;font-size:12px'>{label}: {band}</span>", unsafe_allow_html=True)

def ratio_row(title, node):
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f"**{title}**")
    with col2:
        badge("Band", node.get("band","N/A"))
    st.metric(label="Value", value=fmt_ratio(node.get("value")))
    st.divider()

def build_query_from_ticker(tk: str) -> str:
    tk = (tk or "").strip()
    if not tk:
        return ""
    # run_query는 자연어 입력을 받도록 만들어져 있으므로, 보기 좋은 문장으로 감쌈
    return f"{tk} 유동성/건전성 평가"

# Sidebars
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker (예: AAPL, MSFT, 005930.KS, 7203.T)", value="AAPL")
    show_json = st.toggle("JSON 결과 포함", value=False)
    st.caption("💡 국내/일본/홍콩 종목은 .KS / .T / .HK 접미사를 붙여주세요.")
    run = st.button("분석 시작", type="primary", use_container_width=True)

st.title("📊 Liquidity & Solvency Analyzer")
st.caption("Powered by `finance agent`")

# Main Content Representation 
if run:
    query = build_query_from_ticker(ticker)

    if not query:
        st.warning("상장된 회사의 티커 심볼을 입력해주세요.")
        st.stop()

    # 1) 가능하면 기존 run_query 사용
    result = None
    if hasattr(fa, "run_query"):
        result = fa.run_query(query)
    
    else:
        # run_query가 없다면 최소 폴백: compute만 호출하고 설명은 비움
        payload = fa.compute_ratios_for_ticker(ticker.strip().upper())
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

    core = result["core"]
    ratios = core.get("ratios", {})
    liq = ratios.get("Liquidity", {})
    sol = ratios.get("Solvency", {})

    header_cols = st.columns([3,1,1])
    with header_cols[0]:
        st.subheader(f"{core.get('company','-')}  ({core.get('ticker','-')})")

        if result.get("notes"):
            st.info(result["notes"])

    with header_cols[1]:
        price = core.get("price")
        st.metric("Last Price", value="N/A" if price is None else f"{price:,.2f}")

    with header_cols[2]:
        st.metric("Liquidity Score",
                  value=fmt_ratio(liq.get("current_ratio",{}).get("value")),
                  delta=liq.get("current_ratio",{}).get("band","N/A"))

    st.markdown("### 💧 Liquidity")
    liq_cols = st.columns(3)
    with liq_cols[0]:
        ratio_row("Current Ratio", liq.get("current_ratio",{}))

    with liq_cols[1]:
        ratio_row("Quick Ratio", liq.get("quick_ratio",{}))

    with liq_cols[2]:
        ratio_row("Cash Ratio", liq.get("cash_ratio",{}))

    st.markdown("### 🛡️ Solvency")
    sol_cols = st.columns(3)
    with sol_cols[0]:
        ratio_row("Debt-to-Equity", sol.get("debt_to_equity",{}))

    with sol_cols[1]:
        ratio_row("Debt Ratio", sol.get("debt_ratio",{}))

    with sol_cols[2]:
        ratio_row("Interest Coverage", sol.get("interest_coverage",{}))

    st.markdown("### 📝 Narrative")
    st.write(result.get("explanation", "—"))

    if show_json:
        st.markdown("### Debug JSON")
        st.code(json.dumps(core, indent=2, ensure_ascii=False), language="json")

    st.caption("Note: Ratios are based on the latest available statements and may differ by source. Accuracy Improvement will be updated in the later version.")

else:
    st.info("왼쪽 사이드바에서 티커를 입력하고 **분석 시작**을 눌러보세요.")
