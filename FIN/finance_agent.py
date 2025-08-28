# finance_agent.py
import os
from typing import Dict, Optional, List
import re
import json
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain.tools import tool
from typing import Dict, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.agents import create_openai_functions_agent, AgentExecutor 

load_dotenv()

# LLM (Groq)
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# yfinance Support
def _latest_value_from_df(df: pd.DataFrame, aliases: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    lower_index_map = {str(idx).strip().lower(): idx for idx in df.index}
    try:
        cols_sorted = sorted(df.columns, reverse=True)  # 최신이 좌측인 경우가 많지만 정렬로 견고하게
    except Exception:
        cols_sorted = list(df.columns)
    for alias in aliases:
        for row_lower, row in lower_index_map.items():
            if alias in row_lower:
                series = df.loc[row]
                for c in cols_sorted:
                    val = series.get(c, None)
                    if pd.notnull(val):
                        try:
                            return float(val)
                        except Exception:
                            pass
    return None

def _sum_if_present(*vals: Optional[float]) -> Optional[float]:
    present = [v for v in vals if v is not None]
    return sum(present) if present else None

def compute_ratios_for_ticker(ticker: str) -> dict:
    t = yf.Ticker(ticker.strip())

    ### Balance Sheet, Income Statement, Cashflow Statement DataLoad ###
    # 1. Balance Sheet (분기 → 없으면 연간 폴백)
    q_bs = getattr(t, "quarterly_balance_sheet", None)
    if q_bs is None or getattr(q_bs, "empty", True):
        q_bs = getattr(t, "balance_sheet", None)

    # 2. 손익계산서 (분기 여러 명칭 → 없으면 연간 폴백)
    q_is = getattr(t, "quarterly_financials", None)
    if q_is is None or getattr(q_is, "empty", True):
        q_is = getattr(t, "quarterly_income_stmt", None)
    a_is = getattr(t, "income_stmt", None)  # 연간

    # 3. 현금흐름표 (보조 소스)
    q_cf = getattr(t, "quarterly_cashflow", None)
    a_cf = getattr(t, "cashflow", None)

    # 기본 정보/가격
    info = getattr(t, "info", {}) or {}
    fast = getattr(t, "fast_info", {}) or {}
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector")
    price = None
    try:
        price = float(fast.get("last_price")) if fast else None
    except Exception:
        pass

    # BS가 완전히 비면 조기 반환
    if q_bs is None or getattr(q_bs, "empty", True):
        return {
            "company": company_name,
            "ticker": ticker.upper(),
            "price": price,
            "ratios": {"Liquidity": {}, "Solvency": {}},
            "notes": "해당 티커의 분기/연간 대차대조표를 찾지 못했습니다. 거래소 접미사(.KS, .T, .HK 등) 확인 필요."
        }

    # Helping tools 
    def has_df(df):
        return (df is not None) and (hasattr(df, "empty") and not df.empty)

    def first_non_none(vals):
        for v in vals:
            if v is not None:
                return v
        return None

    def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b in (None, 0):
            return None
        try:
            return float(a) / float(b)
        except Exception:
            return None

    # Balance Sheet에서 항목 추출 
    current_assets = _latest_value_from_df(q_bs, ["total current assets", "current assets"])
    current_liabilities = _latest_value_from_df(q_bs, ["total current liabilities", "current liabilities"])
    inventory = _latest_value_from_df(q_bs, ["inventory"])
    cash = _latest_value_from_df(q_bs, [
        "cash and cash equivalents",
        "cash and cash equivalents including short-term investments",
        "cash and short term investments",
        "cash and short-term investments",
        "cash",
    ])
    short_term_invest = _latest_value_from_df(q_bs, ["short term investments", "short-term investments"])
    cash_like = _sum_if_present(cash, short_term_invest)

    total_assets = _latest_value_from_df(q_bs, ["total assets"])
    total_liabilities = _latest_value_from_df(q_bs, ["total liabilities"])
    equity = _latest_value_from_df(q_bs, ["total stockholder equity", "total shareholders equity", "total equity"])
    short_lt_debt = _latest_value_from_df(q_bs, [
        "short long term debt", "current portion of long term debt", "short-term debt"
    ])
    long_term_debt = _latest_value_from_df(q_bs, ["long term debt"])
    total_debt = _latest_value_from_df(q_bs, ["total debt"]) or _sum_if_present(short_lt_debt, long_term_debt)

    # Income Statement / Cashflow에서 EBIT & 이자비용 추출 
    ebit = first_non_none([
        _latest_value_from_df(q_is, ["ebit", "operating income", "earnings before interest and taxes"]) if has_df(q_is) else None,
        _latest_value_from_df(a_is, ["ebit", "operating income", "earnings before interest and taxes"]) if has_df(a_is) else None,
    ])

    interest_expense = first_non_none([
        _latest_value_from_df(q_is, ["interest expense", "interest expense non operating"]) if has_df(q_is) else None,
        _latest_value_from_df(a_is, ["interest expense", "interest expense non operating"]) if has_df(a_is) else None,
        _latest_value_from_df(q_cf, ["interest paid"]) if has_df(q_cf) else None,  # 보조
        _latest_value_from_df(a_cf, ["interest paid"]) if has_df(a_cf) else None,  # 보조
    ])

    # Ratio Calculation 
    current_ratio = safe_div(current_assets, current_liabilities)
    quick_ratio = safe_div(
        (current_assets - inventory) if (current_assets is not None and inventory is not None) else None,
        current_liabilities
    )
    cash_ratio = safe_div(cash_like, current_liabilities)
    debt_to_equity = safe_div(total_debt, equity)
    debt_ratio = safe_div(total_liabilities, total_assets)

    interest_cov = None
    if ebit is not None and interest_expense is not None:
        try:
            denom = abs(float(interest_expense))
            if denom:
                interest_cov = float(ebit) / denom
        except Exception:
            interest_cov = None

    # Benchmark / Grading 
    def band(val: Optional[float], good: float, fair: float, higher_is_better: bool = True) -> str:
        if val is None:
            return "N/A"
        if higher_is_better:
            if val >= good: return "Strong"
            if val >= fair: return "Fair"
            return "Weak"
        else:
            if val <= good: return "Strong"
            if val <= fair: return "Fair"
            return "Weak"

    assessment = {
        "Liquidity": {
            "current_ratio": {"value": current_ratio, "band": band(current_ratio, 1.5, 1.0, True)},
            "quick_ratio":   {"value": quick_ratio,   "band": band(quick_ratio, 1.0, 0.8, True)},
            "cash_ratio":    {"value": cash_ratio,    "band": band(cash_ratio, 0.5, 0.2, True)},
        },
        "Solvency": {
            "debt_to_equity":    {"value": debt_to_equity, "band": band(debt_to_equity, 1.0, 2.0, False)},
            "debt_ratio":        {"value": debt_ratio,     "band": band(debt_ratio, 0.5, 0.7, False)},
            "interest_coverage": {"value": interest_cov,   "band": band(interest_cov, 5.0, 2.0, True)},
        }
    }

    return {
        "company": company_name,
        "ticker": ticker.upper(),
        "sector": sector,
        "price": price,
        "ratios": assessment,
        #"raw": {
            #"current_assets": current_assets,
            #"current_liabilities": current_liabilities,
            #"inventory": inventory,
            #"cash_like": cash_like,
            #"total_assets": total_assets,
            #"total_liabilities": total_liabilities if (total_liabilities is not None) else None,
            #"equity": equity,
            #"total_debt": total_debt,
            #"ebit": ebit,
            #"interest_expense": interest_expense,
        #},
        "notes": "Latest quarterly (fallback to annual) statements via yfinance; ratios are approximations."
    }

def explain_ratios_with_llm(ratio_payload: Dict, language: str = "ko") -> str:
    """
    계산된 지표(JSON)를 LLM에 넘겨 요약/판단문을 생성.
    """
    ask_lang = "Korean" if language.lower().startswith("ko") else "English"
    prompt = (
        f"You are a financial analysis assistant. Write in {ask_lang}.\n"
        f"- Given the JSON payload with liquidity and solvency ratios,\n"
        f"  produce a concise assessment with bullets.\n"
        f"- Show key ratios with 2 decimals and label (Strong/Fair/Weak).\n"
        f"- Add one plain-language takeaway.\n\n"
        f"JSON:\n{ratio_payload}"
    )
    res = llm.invoke(prompt)
    # content extraction 
    return getattr(res, "content", str(res))

# Tool 1
class RatiosOutput(BaseModel):
    company: str
    ticker: str
    sector: Optional[str] = None
    price: Optional[float] = Field(None, description="Last price if available")
    ratios: Dict
    notes: Optional[str] = None

def extract_ticker(user_query: str) -> str:
    """
    문장에서 주식 티커처럼 보이는 토큰을 뽑는다.
    - 영문/숫자/.- 만 허용
    - 첫 번째 유효 후보를 선택 (마지막 토큰이 아닌!)
    - 없으면 원문 대문자 반환
    """
    # 대문자로 변환 후 티커 후보 전부 추출
    tokens = re.findall(r"[A-Z0-9\.\-]{1,15}", user_query.upper())
    # 완전 숫자만인 토큰은 제외(보통 티커는 영문 포함, 다만 일부 거래소 예외는 허용 가능)
    candidates = [t for t in tokens if any(ch.isalpha() for ch in t)]
    return candidates[0] if candidates else user_query.upper()

def pick_valid_ticker(user_query: str) -> str:
    """
    문장에서 티커 후보들을 추출하고, yfinance로 검증해
    '재무제표가 실제로 존재하는' 첫 번째 심볼을 반환.
    없으면 첫 후보를, 그마저 없으면 원문 대문자를 반환.
    """
    # AAPL, MSFT, 005930.KS, 7203.T, 0700.HK, BRK.B 같은 패턴 허용
    tokens = re.findall(r"[A-Za-z0-9\.\-]{1,15}", user_query.upper())
    # 영문자가 1개 이상 포함된 토큰만 우선 (전부 숫자인 코드 제외)
    candidates = [t for t in tokens if any(c.isalpha() for c in t)]
    if not candidates:
        return user_query.upper().strip()

    # 후보들을 검증: balance sheet가 비어있지 않은 심볼이면 OK
    for sym in candidates:
        try:
            t = yf.Ticker(sym.strip())
            bs = t.quarterly_balance_sheet
            if isinstance(bs, pd.DataFrame) and not bs.empty:
                return sym.strip()
        except Exception:
            pass

    # 그래도 못 찾으면 첫 후보 반환 (AAPL LIQUIDITY -> 'AAPL')
    return candidates[0].strip()

# Tool 2
@tool("finance_health_tool")
def finance_health_tool(ticker: str) -> str:
    """티커를 받아 유동성/건전성 비율을 계산해 JSON 문자열로 반환."""
    data = compute_ratios_for_ticker(ticker)
    return json.dumps(data, ensure_ascii=False)

# Prompt (⚠️ use ChatPromptTemplate)
SYSTEM_PROMPT = (
    "You are a financial analysis assistant. "
    "When the user provides a stock ticker, call the finance_health_tool to get data, "
    "then explain liquidity (current/quick/cash) and solvency (D/E, debt ratio, interest coverage). "
    "Be concise, show key ratios with 2 decimals, and give plain-language diagnosis."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

#agent = create_openai_functions_agent(llm, [finance_health_tool], prompt)
#executor = AgentExecutor(agent=agent, tools=[finance_health_tool], verbose=True)

def run_query(user_query: str) -> dict:
    ticker = pick_valid_ticker(user_query)  # ✅ 새 함수 사용
    payload = compute_ratios_for_ticker(ticker)

    # (선택) 비어있으면 안내 메시지
    if (not payload.get("ratios")
        or not payload["ratios"].get("Liquidity")
        or not payload["ratios"].get("Solvency")
        or all(v.get("value") is None for v in payload["ratios"]["Liquidity"].values())
        and all(v.get("value") is None for v in payload["ratios"]["Solvency"].values())):
        payload["notes"] = (
            f"'{ticker}' 재무제표를 찾지 못했습니다. 거래소 접미사(.KS, .T, .HK 등) 필요 여부를 확인하세요."
        )
        explanation = "재무제표가 비어 있어 평가를 생성하지 않았습니다."
    else:
        explanation = explain_ratios_with_llm(payload, language="ko")

    return {
        "core": {
            "company": payload["company"],
            "ticker": payload["ticker"],
            "price": payload["price"],
            "ratios": payload["ratios"],
        },
        "notes": payload.get("notes"),
        "explanation": explanation,
    }

if __name__ == "__main__":
    q = input("예: 'AAPL 유동성/건전성 평가' > ").strip()
    out = run_query(q)

    #print("\n--- Core Data ---\n")
    #print(json.dumps(out["core"], indent=2, ensure_ascii=False))
    
    # Final Result Delievered 
    print("\n--- Explanation ---\n")
    print(out["explanation"])