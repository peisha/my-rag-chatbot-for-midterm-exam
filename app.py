# app.py — KBS 한국어능력시험 RAG 튜터 (어휘/규정/다의어 + 파일기반 RAG + 퀴즈)
import os
import json
import random
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) 환경변수 로드 (.env 또는 Streamlit Secrets → 환경변수로 설정되어 있어야 함)
load_dotenv(override=True)

# 2) Streamlit 기본 설정
st.set_page_config(page_title="KBS 한국어능력시험 RAG 튜터", layout="wide")
# 제목 (두 줄)
st.markdown(
    "<h1 style='text-align:center; line-height:1.3;'>"
    "✨😎 오로지 당신만을 위한~!<br>"
    "KBS 한국어능력시험 쌤 💕💫"
    "</h1>",
    unsafe_allow_html=True
)

# 제목과 안내문 사이 1줄 공백
st.markdown("<br>", unsafe_allow_html=True)

# 안내문 (두 줄로 줄바꿈 적용)
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:14px; margin:10px 0 24px;'>"
    "규정 근거는 국립국어원의 '한글맞춤법·표준발음법·외래어·로마자 표기법' 및 "
    "'표준국어대사전'에 있습니다."
    "</p>",
    unsafe_allow_html=True
)

# ───────── 데이터 로더 (어휘/규정/다의어)
@st.cache_data
def load_lexicon_df():
    """
    통합 어휘 로더: 고유어/관용구/속담/사자성어/순화어
    각 CSV 스키마: [유형, 어휘, 뜻풀이] (예문, 비고는 선택)
    """
    files = [
        "data/고유어.csv", "data/관용구.csv", "data/속담.csv",
        "data/사자성어.csv", "data/순화어.csv",
    ]
    cols_base = ["유형", "어휘", "뜻풀이"]
    dfs = []
    for p in files:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for c in cols_base:
                if c not in df.columns: df[c] = ""
            for c in ["예문", "비고"]:
                if c not in df.columns: df[c] = ""
            dfs.append(df[cols_base + ["예문", "비고"]])
    if not dfs:
        return pd.DataFrame(columns=cols_base + ["예문", "비고"])
    out = pd.concat(dfs, ignore_index=True)
    for c in out.columns:
        out[c] = out[c].fillna("").astype(str)
    return out

@st.cache_data
def load_rules_list():
    # TODO: 규정 데이터 로딩 함수 구현 예정
    return []

@st.cache_data
def load_poly_df():
    """data/polysemy.csv (표제어,의미번호,뜻,예문) → 항상 DataFrame 반환"""
    path = "data/polysemy.csv"
    cols = ["표제어","의미번호","뜻","예문"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    # 누락 컬럼 보정
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].fillna("").astype(str)

# CSV 파일 로드
VOCAB = load_lexicon_df()
RULES = load_rules_list() or []
POLY = load_poly_df()

# ==========================
# 1) 규정 JSON 로드 & 문서화
# ==========================
# 상단 import 근처
import json, os
from langchain_core.documents import Document
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

RULES_JSON_PATH = "rules.json"  # 업로드로 덮어쓸 파일명

@st.cache_data(show_spinner=False)
def load_rule_docs(path: str = RULES_JSON_PATH) -> list[Document]:
    """rules.json → LangChain Document 목록. 파일 없거나 파싱 실패 시 []"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    docs: list[Document] = []
    for row in data:
        j = row.get
        장, 절, 항, 제목 = j("장",""), j("절",""), j("항",""), j("제목","")
        설명 = j("설명","")
        ok   = ", ".join(j("예시_옳음",  []))
        bad  = ", ".join(j("예시_틀림", []))
        excp = ", ".join(j("예시_예외",  []))
        body = [설명]
        if ok:  body.append(f"[예시_옳음] {ok}")
        if bad: body.append(f"[예시_틀림] {bad}")
        if excp: body.append(f"[예시_예외] {excp}")
        text = f"{장} · {절} · {항}\n{제목}\n\n" + "\n".join(body)
        docs.append(Document(page_content=text, metadata={"장": 장, "절": 절, "항": 항, "제목": 제목}))
    return docs

# =========================
# 2) 벡터 스토어 구축/로드
# =========================
@st.cache_resource(show_spinner=False)
def build_rule_retriever(docs: List[Document]):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embed)
    return vs.as_retriever(search_kwargs={"k": 4})

# 부팅 시 규정 색인 준비 (파일 없으면 None)
rule_docs = load_rule_docs(RULES_JSON_PATH)
retriever_rules = build_rule_retriever(rule_docs) if rule_docs else None

# =========================
# 3) 규정 Q&A 함수
# =========================
def answer_rule(user_q: str) -> str:
    """규정 전용 질의 응답: 최상위 근거 1~2개를 요약해 보여줌"""
    hits = rule_retriever.invoke(user_q)
    if not hits:
        return "규정에서 관련 내용을 찾지 못했어요. 질문을 조금만 바꿔서 다시 시도해 주세요!"

    # 최상위 결과로 간단한 답변 + 메타(장/절/항/제목)와 근거 일부를 보여줌
    top = hits[0]
    meta = top.metadata
    header = f"**{meta.get('장','')} · {meta.get('절','')} · {meta.get('항','')}** — {meta.get('제목','')}"
    snippet = top.page_content[:800]  # 너무 길면 800자까지만
    out = f"{header}\n\n{snippet}"

    # 추가 근거(선택)
    if len(hits) > 1:
        out += "\n\n---\n**추가 근거**\n"
        for h in hits[1:3]:
            m = h.metadata
            out += f"- {m.get('장','')} {m.get('절','')} {m.get('항','')} — {m.get('제목','')}\n"

    return out

# =========================
# 4) (선택) 앱에서 파일 업로드로 교체 가능
# =========================
with st.expander("📤 규정 JSON 업로드 (선택)"):
    up = st.file_uploader("rules.json 업로드", type=["json"])
    if up is not None:
        with open(RULES_JSON_PATH, "wb") as f:
            f.write(up.read())
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("규정 데이터가 갱신되었어요. 잠시 후 자동으로 반영됩니다.")

# ─────────────────────────────────────────────────────────────
# 파일 업로드 → 텍스트 추출 → 벡터DB 구성 (업로드시만)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <p style='text-align:center; font-size:15px; color:#4b5563; margin-top:10px; margin-bottom:14px;'>
        📂 별도로 학습시키고 싶은 자료가 있으시다면<br>
        <span style='color:#111827; font-weight:600;'>하단에 첨부해주세요.</span>
    </p>
    """,
    unsafe_allow_html=True
)

uploaded = st.file_uploader("이곳에 txt 또는 pdf 파일을 업로드하세요", type=["txt", "pdf"])

def load_text(file):
    if file is None:
        return ""
    if file.type == "text/plain":
        return file.read().decode("utf-8", errors="ignore")
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    st.error("지원하지 않는 형식입니다.")
    st.stop()

text = ""
vectordb = None
retriever = None

if uploaded:
    text = load_text(uploaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n","\n"," "]
    )
    docs = splitter.create_documents([text])

    with st.spinner("임베딩·색인 구성 중… (최초 1회는 모델 다운로드로 다소 소요)"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = FAISS.from_documents(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# ─────────────────────────────────────────────────────────────
# LLM (생성)
# ─────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY가 설정되어 있지 않아요. (Streamlit Secrets 또는 .env 설정 필요)")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "다음 컨텍스트를 참고해서 한국어로 간결하고 정확히 답변하세요.\n"
    "가능하면 근거(예문/규정/출전)을 함께 제시하세요.\n\n"
    "컨텍스트:\n{context}\n\n"
    "질문: {question}"
)

def run_rag(question: str) -> str:
    """업로드 자료 기반 RAG (자료 없으면 빈 컨텍스트)"""
    if retriever is None:
        context = ""
    else:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# ─────────────────────────────────────────────────────────────
# 간단 의도 분류 → (어휘/규정/다의어/파일 RAG) 라우팅
# ─────────────────────────────────────────────────────────────
def intent(text: str) -> str:
    t = text.strip()
    if any(k in t for k in ["맞춤법","띄어쓰기","발음","외래어","로마자","표기","옳은 표기","맞나요"]):
        return "rule"
    if any(k in t for k in ["다의어","여러 뜻","뜻들","의미들"]):
        return "poly"
    if any(k in t for k in ["뜻","의미","사자성어","속담","유의어","관용구","단어"]):
        return "vocab"
    return "rag"

def answer_vocab(q: str) -> str:
    if VOCAB.empty:
        return "사전 데이터(고유어·관용구·속담·사자성어·순화어 CSV)가 아직 없습니다."
    hit = VOCAB[VOCAB["어휘"].apply(lambda w: isinstance(w, str) and w in q)]
    if len(hit):
        row = hit.iloc[0]
        lines = [
            f"〔{row.get('유형','어휘')}〕 {row.get('어휘','-')}",
            f"뜻: {row.get('뜻풀이','-')}",
        ]
        ex = row.get("예문","")
        if isinstance(ex, str) and ex.strip():
            lines.append(f"예문: {ex}")
        extra = row.get("비고","")
        if isinstance(extra, str) and extra.strip():
            lines.append(f"비고: {extra}")
        return "\n".join(lines)
    back = run_rag(f"어휘 의미: {q}")
    return "사전에 직접 일치하는 어휘가 없어요.\n\n" + back


def answer_rule(q: str):
    """규정 전용 RAG: rules.json 기반으로 검색"""
    if retriever_rules is None:
        return "규정 데이터가 아직 로드되지 않았어요. rules.json을 업로드해 주세요!"

    hits = retriever_rules.invoke(q)
    if not hits:
        return "관련 규정을 찾지 못했어요. 질문을 조금 다르게 써 볼까요?"

    top = hits[0]
    meta = top.metadata
    header = f"**{meta.get('장','')} · {meta.get('절','')} · {meta.get('항','')} — {meta.get('제목','')}**"
    snippet = top.page_content[:800]
    out = f"{header}\n\n{snippet}"

    if len(hits) > 1:
        out += "\n\n---\n**추가 근거**\n"
        for h in hits[1:3]:
            m = h.metadata
            out += f"- {m.get('장','')} {m.get('절','')} {m.get('항','')} — {m.get('제목','')}\n"
    return out


def answer_poly(q: str) -> str:
    if POLY.empty:
        return "다의어 데이터(polysemy.csv)가 아직 없습니다."
    cands = [w for w in POLY["표제어"].unique() if isinstance(w, str) and w in q]
    if not cands:
        return "어떤 단어의 여러 뜻을 묻는지 알려 주세요. (예: '들다 다의어 알려줘')"
    w = max(cands, key=len)
    rows = POLY[POLY["표제어"] == w]
    lines = [f"‘{w}’의 뜻"]
    for _, r in rows.sort_values("의미번호").iterrows():
        num = int(r["의미번호"]) if str(r["의미번호"]).isdigit() else r["의미번호"]
        lines.append(f" {num}. {r['뜻']}  (예: {r['예문']})")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# 퀴즈 문항 생성기: vocab.csv에서 n문항 뽑기 (메타는 [유형]만)
# ─────────────────────────────────────────────────────────────
import re

def clean_text(s: str) -> str:
    """보기/질문 앞뒤의 특수문자나 숫자 제거"""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # '=', '-', '⑥', '①'~'⑩', 괄호, 숫자 등 제거
    s = re.sub(r'^[=\-\d⑴-⑽⑴⑽\(\)\s·\.\,]+', '', s)
    s = re.sub(r'[\s·\.\,]+$', '', s)
    return s.strip()

# ==== 공통: CSV 안전 로더 ====
def _read_csv_expect(path: str, expected_cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    for c in expected_cols:
        if c not in df.columns: df[c] = ""
    return df[expected_cols].fillna("").astype(str)

# ==== 통합 어휘 퀴즈 ====
def build_quiz_lexicon(df: pd.DataFrame, n: int) -> list[dict]:
    need = {"유형","어휘","뜻풀이"}
    if not need.issubset(df.columns) or df.empty: return []
    base = df.dropna(subset=["어휘","뜻풀이"]).copy().sample(frac=1.0)
    items = []
    for _, r in base.iterrows():
        q_word, correct = r["어휘"].strip(), r["뜻풀이"].strip()
        cat, ex = r.get("유형","어휘"), str(r.get("예문","")).strip()
        if not q_word or not correct: continue
        same = base[(base["유형"]==cat) & (base["뜻풀이"]!=correct)]["뜻풀이"].unique().tolist()
        if len(same) < 3:
            same = base[base["뜻풀이"]!=correct]["뜻풀이"].unique().tolist()
        if len(same) < 3: continue
        choices = random.sample(same, 3) + [correct]
        random.shuffle(choices)
        items.append({"question": f"‘{q_word}’의 뜻으로 가장 알맞은 것은?",
                      "choices": choices, "answer": correct, "ex": ex, "meta": f"[{cat}]"})
        if len(items) >= n: break
    return items

# ==== 띄어쓰기 ====
def build_quiz_spacing(n: int) -> list[dict]:
    df = _read_csv_expect("data/띄어쓰기.csv", ["유형","정답","오답"])
    items = []
    if df.empty: return items
    for _, r in df.sample(frac=1.0).iterrows():
        wrong, correct, cat = r["오답"], r["정답"], r.get("유형","띄어쓰기")
        if not wrong or not correct: continue
        other = df[df["정답"]!=correct]["정답"].unique().tolist()
        if len(other) < 3: continue
        choices = random.sample(other, 3) + [correct]
        random.shuffle(choices)
        items.append({"question": f"‘{wrong}’의 올바른 띄어쓰기는?",
                      "choices": choices, "answer": correct, "ex": "", "meta": f"[{cat}]"})
        if len(items) >= n: break
    return items

# ==== 맞춤법 ====
def build_quiz_orthography(n: int) -> list[dict]:
    df = _read_csv_expect("data/맞춤법.csv", ["유형","정답 단어","오답 단어"])
    items = []
    if df.empty: return items
    rights = df["정답 단어"].unique().tolist()
    for _, r in df.sample(frac=1.0).iterrows():
        wrong, correct, cat = r["오답 단어"], r["정답 단어"], r.get("유형","맞춤법")
        if not wrong or not correct: continue
        other = [x for x in rights if x != correct]
        if len(other) < 3: continue
        choices = random.sample(other, 3) + [correct]
        random.shuffle(choices)
        items.append({"question": f"‘{wrong}’의 올바른 표기는?",
                      "choices": choices, "answer": correct, "ex": "", "meta": f"[{cat}]"})
        if len(items) >= n: break
    return items

# ==== 외래어 ====
def build_quiz_loanword(n: int) -> list[dict]:
    df = _read_csv_expect("data/외래어.csv", ["유형","정답 외래어","오답 외래어"])
    items = []
    if df.empty: return items
    rights = df["정답 외래어"].unique().tolist()
    for _, r in df.sample(frac=1.0).iterrows():
        wrong, correct, cat = r["오답 외래어"], r["정답 외래어"], r.get("유형","외래어")
        if not wrong or not correct: continue
        other = [x for x in rights if x != correct]
        if len(other) < 3: continue
        choices = random.sample(other, 2) + [wrong] + [correct]
        random.shuffle(choices)
        items.append({"question": "다음 중 올바른 외래어 표기는?",
                      "choices": choices, "answer": correct, "ex": "", "meta": f"[{cat}]"})
        if len(items) >= n: break
    return items

# ==== 표준발음법 ====
def build_quiz_pron(n: int) -> list[dict]:
    df = _read_csv_expect("data/표준발음법.csv", ["유형","단어","표준 발음"])
    items = []
    if df.empty: return items
    for _, r in df.sample(frac=1.0).iterrows():
        word, correct, cat = r["단어"], r["표준 발음"], r.get("유형","표준발음법")
        if not word or not correct: continue
        other = df[df["표준 발음"]!=correct]["표준 발음"].unique().tolist()
        if len(other) < 3: continue
        choices = random.sample(other, 3) + [correct]
        random.shuffle(choices)
        items.append({"question": f"‘{word}’의 표준 발음은?",
                      "choices": choices, "answer": correct, "ex": "", "meta": f"[{cat}]"})
        if len(items) >= n: break
    return items

# ==== 로마자표기법 ====
def build_quiz_romaja(n: int) -> list[dict]:
    df = _read_csv_expect("data/로마자표기법.csv", ["유형","단어","로마자"])
    items = []
    if df.empty: return items
    for _, r in df.sample(frac=1.0).iterrows():
        word, correct, cat = r["단어"], r["로마자"], r.get("유형","로마자표기법")
        if not word or not correct: continue
        other = df[df["로마자"]!=correct]["로마자"].unique().tolist()
        if len(other) < 3: continue
        choices = random.sample(other, 3) + [correct]
        random.shuffle(choices)
        items.append({"question": f"‘{word}’의 로마자 표기는?",
                      "choices": choices, "answer": correct, "ex": "", "meta": f"[{cat}]"})
        if len(items) >= n: break
    return items

# ==== 표준어규정 (유형/단어) ====
def build_quiz_standard_rule(n: int) -> list[dict]:
    df = _read_csv_expect("data/표준어규정.csv", ["유형","단어"])
    items = []
    if df.empty: return items
    kinds = df["유형"].dropna().unique().tolist()
    if len(kinds) < 4 and kinds:
        while len(kinds) < 4: kinds.append(random.choice(kinds))
    for _, r in df.sample(frac=1.0).iterrows():
        word, correct = r["단어"], r["유형"]
        if not word or not correct: continue
        wrongs = [k for k in kinds if k != correct]
        if len(wrongs) < 3: continue
        choices = random.sample(wrongs, 3) + [correct]
        random.shuffle(choices)
        items.append({"question": f"‘{word}’은(는) 어떤 표준어 규정에 해당할까요?",
                      "choices": choices, "answer": correct, "ex": "", "meta": "[표준어규정]"})
        if len(items) >= n: break
    return items

# ==== 전체 혼합 퀴즈 ====
def build_all_quiz_items(total: int = 10) -> list[dict]:
    per = max(1, total // 6)   # 대략 균등
    items = []
    items += build_quiz_lexicon(VOCAB, n=per*2)   # 어휘 비중 조금 더
    items += build_quiz_spacing(per)
    items += build_quiz_orthography(per)
    items += build_quiz_loanword(per)
    items += build_quiz_pron(per)
    items += build_quiz_romaja(per)
    items += build_quiz_standard_rule(per)
    random.shuffle(items)
    return items[:total]

# ─────────────────────────────────────────────────────────────
# 탭 UI: 시험 소개 · 질문하기 | 퀴즈 | 학습하기 | 오답노트
# ─────────────────────────────────────────────────────────────
tab_intro_ask, tab_quiz, tab_learn, tab_wrong = st.tabs(
    ["🥰 시험 소개 · 🧐 질문하기", "🤗 퀴즈 풀기", "📚 학습하기", "📘 오답노트"]
)

# ───────── 1) 시험 소개 · 질문하기 탭 ─────────
with tab_intro_ask:
    # ── (A) 시험 소개 섹션 ──
    st.markdown("<h2>📘 KBS 한국어능력시험 소개</h2>", unsafe_allow_html=True)

    with st.expander("🌾 한능시! 왜 필요한가요?", expanded=True):
        st.markdown(
            """
**입학 / 취업 / 승진에 유리합니다.**
- 특목고 진학 및 대학 입학  
- KBS·EBS·경향신문 등 언론사 취업  
- 우리은행·GS홈쇼핑 등 민간기업 취업  
- 건강보험공단·한국전력 등 공공기관 취업  
- 경찰청 등 승진에서 ‘결정적 가산점’ 확보
            """
        )

    # ── (B) 시험 구성 섹션 ──
    with st.expander("🍹 시험 구성 및 영역", expanded=True):
        # 설명 문장
        st.markdown(
            """
본 시험은 객관식 **5지 선다형**으로 구성되어 있으며, 총 **100문항**입니다.  
아래는 시간대별 영역 구성표입니다.
            """
        )

        # 표 (색상 제거 + 줄바꿈 반영)
        st.markdown(
            """
<style>
.exam-table {
  border-collapse: collapse;
  width: 100%;
  margin-top: 10px;
  font-size: 16px;
  text-align: center;
}
.exam-table th {
  background-color: #f8f9fa;
  padding: 10px;
  border-bottom: 2px solid #ddd;
}
.exam-table td {
  padding: 10px;
  border-bottom: 1px solid #eee;
}
</style>

<table class="exam-table">
  <tr>
    <th>시간대</th>
    <th>영역</th>
    <th>문항 수</th>
  </tr>
  <tr>
    <td>10:00~10:25<br>(25분)</td>
    <td>듣기·말하기</td>
    <td>15문항</td>
  </tr>
  <tr>
    <td>10:25~12:00<br>(95분)</td>
    <td>쓰기 · 창안 · 읽기 · 국어문화<br>· 어휘 · 어법</td>
    <td>85문항<br>(쓰기 5 + 창안 10 + 읽기 30 + 국어문화 10 + 어휘·어법 30)</td>
  </tr>
</table>
            """,
            unsafe_allow_html=True,
        )

    # ── (C) 영역별 출제 경향 ──
    with st.expander("🍡 영역별 출제 경향"):
        st.markdown(
            """
<div style="font-size:15px; line-height:1.8;">
  <p><b>🍫 듣기·말하기</b></p>
  <ul style="margin-left:12px;">
     ✔️ 그림·장면·라디오 듣고 핵심 내용 파악
     ✔️ 고전/우화/시 청취 후 의미 및 감정 추론
     ✔️ 대화·발표 듣고 말하기 방식·화자의 태도 파악
  </ul>

  <p><b>🍮 어휘</b></p>
  <ul style="margin-left:12px;">
     ✔️ 고유어·한자어 뜻/표기 구분
     ✔️ 어휘 관계(유의/반의/상하)·속담·관용구
     ✔️ 외래어·한자어의 올바른 우리말 쓰임
  </ul>

  <p><b>🍰 어법</b></p>
  <ul style="margin-left:12px;">
     ✔️ 맞춤법/표준어/발음/표기법 구분
     ✔️ 문장 호응 및 잘못된 표현 진단·수정
  </ul>

  <p><b>🍯 쓰기</b></p>
  <ul style="margin-left:12px;">
     ✔️ 계획·개요 수정, 자료 활용, 글 고쳐쓰기
  </ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    # ── (B) 질문하기 섹션 ──
    st.markdown(
        "<p style='font-size:18px; font-weight:600;'>🐲 저에게 질문을 해주세용 🌟!</p>"
        "<p style='font-size:14px; color:#555; margin-top:-10px; margin-bottom:-8px;'>"
        "(e.g. '교각살우'의 뜻이 궁금해요,<br>"
        "'늑막염'의 표준 발음을 알려주세요,<br>"
        "자료에서 ~ 는 어디에 나오나요?)"
        "</p>",
        unsafe_allow_html=True
    )

    # 입력창 (예시 문구와 딱 붙게)
    user_q = st.text_input("", placeholder="여기에 질문을 입력하세요 😊")

    # 응원 문구 (2줄 공백 후)
    st.markdown(
        "<br><br>"
        "<p style='color:gray; font-size:20px; font-weight:600; text-align:center;'>"
        "파이팅!!! 쌤은 여러분의 시험을 응원합니다~! 🥰"
        "</p>",
        unsafe_allow_html=True
    )

    if user_q:
        kind = intent(user_q)
        with st.spinner("답변 생성 중…"):
            try:
                if kind == "vocab":
                    ans = answer_vocab(user_q)
                elif kind == "rule":
                    ans = answer_rule(user_q)
                elif kind == "poly":
                    ans = answer_poly(user_q)
                else:
                    ans = run_rag(user_q)

                st.write(ans)

                if kind == "rag" and retriever is not None:
                    with st.expander("🔎 근거 보기(업로드 자료에서 추출)"):
                        docs = retriever.invoke(user_q)
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**근거 {i}**")
                            st.code((d.page_content or "")[:800])
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# 나머지 탭(tab_quiz, tab_learn, tab_wrong)은
# 기존에 쓰던 코드 블록을 그대로 이어서 사용하세요.

# ─────────────────────────────────────────────────────────────
# 퀴즈 탭 (확장 버전: 제출 → 결과 → 새 퀴즈 버튼 순서)
# ─────────────────────────────────────────────────────────────
with tab_quiz:
    st.markdown("❤️쨔란! **랜덤 종합 퀴즈**를 풀어보세요!😘")

    # 처음 로드 시 data/ 모든 CSV 기반으로 혼합 퀴즈 생성
    if "quiz_items" not in st.session_state or not st.session_state.quiz_items:
        st.session_state.quiz_items = build_all_quiz_items(total=10)
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = 0

    if not st.session_state.quiz_items:
        st.info("퀴즈를 만들 데이터가 부족합니다. data/ 폴더의 CSV들을 확인해 주세요.")
    else:
        # 문항 렌더링
        answers = {}
        for i, item in enumerate(st.session_state.quiz_items):
            st.markdown(
                f"**Q{i+1}. {item['question']}**  \n<small>{item['meta']}</small>",
                unsafe_allow_html=True
            )
            key = f"quiz_q_{i}"
            choice = st.radio(
                "보기",
                options=[clean_text(c) for c in item["choices"]],
                index=None,
                key=key,
                label_visibility="collapsed"
            )
            answers[i] = choice
            st.divider()

        # 버튼 1세트만 배치
        submitted = st.button("✅ 제출", key="quiz_submit_btn", type="primary", use_container_width=True)
        new_quiz  = st.button("🔄 새 퀴즈 출제", key="quiz_new_btn", use_container_width=True)

        # 제출 → 채점/오답노트 저장/해설
        if submitted:
            score = 0
            results = []
            wrong_items = []

            for i, item in enumerate(st.session_state.quiz_items):
                sel = answers.get(i)
                ok = (sel == item["answer"])
                score += int(ok)
                results.append((ok, sel, item))
                if not ok:
                    wrong_items.append({
                        "문항": item["question"],
                        "선택한 답": sel if sel is not None else "(무응답)",
                        "정답": item["answer"],
                        "예문": item.get("ex", "")
                    })

            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            st.session_state["wrong_items"] = wrong_items

            st.success(f"점수: **{score} / {len(st.session_state.quiz_items)}**")
            with st.expander("정답 및 해설 보기"):
                for i, (ok, sel, item) in enumerate(results, start=1):
                    icon = "✅" if ok else "❌"
                    sel_txt = sel if sel is not None else "(무응답)"
                    st.markdown(f"**{icon} Q{i}. {item['question']}**")
                    st.write(f"- 선택: {sel_txt}")
                    st.write(f"- 정답: {item['answer']}")
                    if item.get("ex"):
                        st.write(f"- 예문: {item['ex']}")
                    st.write("---")

        # 새 퀴즈
        if new_quiz:
            st.session_state.quiz_items = build_all_quiz_items(total=10)  # 총 문항 수
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.rerun()

# ========== 학습 탭 헬퍼 ==========
def init_study():
    """학습 진행 상태 초기화"""
    if "study" not in st.session_state:
        st.session_state.study = {
            "today_goal": {"lex": 10, "rule": 5, "poly": 3},
            "seen_ids": [],
            "bookmarks": [],
            "leitner": {"1": [], "2": [], "3": []},
            "progress": {
                "date": pd.Timestamp.today().date().isoformat(),
                "lex": 0, "rule": 0, "poly": 0
            }
        }

@st.cache_data
def rules_df():
    """rules.json → DataFrame"""
    return pd.DataFrame(RULES) if RULES else pd.DataFrame(columns=["규정명","항목","설명","예시"])

def show_rule_card(idx: int) -> int:
    """규정 학습용 카드 뷰 + 이전/다음/북마크"""
    df = rules_df()
    if df.empty:
        st.info("rules.json이 비어 있습니다.")
        return 0
    idx = max(0, min(idx, len(df)-1))
    r = df.iloc[idx]
    st.markdown(f"### 〔{r.get('규정명','규정')}〕 {r.get('항목','')}")
    st.write(r.get("설명",""))
    ex = r.get("예시","")
    if isinstance(ex, str) and ex.strip():
        st.code(ex)
    c1, c2, c3 = st.columns(3)
    if c1.button("⬅️ 이전", key=f"rule_prev_{idx}"):
        idx -= 1
    if c2.button("⭐ 중요 표시", key=f"rule_star_{idx}"):
        st.session_state.study["bookmarks"].append(("rule", idx))
        st.toast("중요 항목으로 저장했어요!", icon="⭐")
    if c3.button("다음 ➡️", key=f"rule_next_{idx}"):
        idx += 1
    return idx

def flash_lex(df: pd.DataFrame):
    """어휘 플래시카드: 앞(어휘) / 뒤(뜻풀이·예문)"""
    if df.empty:
        st.info("어휘 CSV가 비어 있습니다.")
        return
    if "lex_idx" not in st.session_state:
        st.session_state.lex_idx = 0
    i = st.session_state.lex_idx % len(df)
    row = df.iloc[i]
    front = f"**〔{row.get('유형','어휘')}〕 {row.get('어휘','(어휘)')}**"
    ex = row.get("예문","")
    back_lines = [f"뜻: {row.get('뜻풀이','-')}"]
    if isinstance(ex, str) and ex.strip():
        back_lines.append(f"\n예문: {ex}")
    back = "\n".join(back_lines)

    st.markdown(front)
    if st.toggle("정답 보기", key=f"lex_show_{i}"):
        st.write(back)

    c1, c2, c3 = st.columns(3)
    if c1.button("틀림", key=f"lex_wrong_{i}"):
        st.session_state.study["leitner"]["1"].append(("lex", i))
        st.session_state.lex_idx += 1
    if c2.button("정답", key=f"lex_right_{i}"):
        st.session_state.study["progress"]["lex"] += 1
        st.session_state.study["leitner"]["2"].append(("lex", i))
        st.session_state.lex_idx += 1
    if c3.button("건너뛰기", key=f"lex_skip_{i}"):
        st.session_state.lex_idx += 1

# ─────────────────────────────────────────────────────────────
# 📚 학습하기 탭
# ─────────────────────────────────────────────────────────────
with tab_learn:
    init_study()
    S = st.session_state.study
    G = S["today_goal"]; P = S["progress"]

    # ── 규정 학습 ──
    with st.expander("🧋 규정 학습 🥂", expanded=True):
        if "rule_idx" not in st.session_state:
            st.session_state.rule_idx = 0

        # 규정 카드 보여주기 (Ellipsis 출력 원인 제거: 절대 빈 블록/… 사용 X)
        st.session_state.rule_idx = show_rule_card(st.session_state.rule_idx)

        if st.button("학습 완료(규정 1 증가)", key="rule_done_learn"):
            st.session_state.study["progress"]["rule"] += 1
            st.toast("규정 1개 학습 완료!", icon="✅")

    # ── 규정 JSON 업로드 (선택) ──
    with st.expander("🍈 규정 JSON 업로드 (선택사항입니다) 🌽"):
        up = st.file_uploader(
            "rules.json 업로드",
            type=["json"],
            key="rules_json_uploader_learn"
        )
        if up is not None:
            with open(RULES_JSON_PATH, "wb") as f:
                f.write(up.read())
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("규정 데이터가 새로 적용되었어요. 잠시 후 자동으로 반영됩니다.")
            st.rerun()

    # ── 어휘 플래시카드 ──
    with st.expander("🍇 어휘 플래시카드 🍓", expanded=True):
        flash_lex(VOCAB)

    # 미니 테스트 (방금 학습한 맥락으로 5문항)
    with st.expander("🍋 미니 테스트 (방금 학습한 맥락으로 5문항!) 🫐"):
        mini_items = build_all_quiz_items(total=5)  # 기존 빌더 재사용
        mini_answers = {}
        for i, q in enumerate(mini_items):
            st.markdown(f"**Q{i+1}. {q['question']}**  \n<small>{q['meta']}</small>", unsafe_allow_html=True)
            mini_answers[i] = st.radio("보기", q["choices"], index=None, key=f"mini_{i}", label_visibility="collapsed")
            st.divider()
        if st.button("채점", key="mini_grade_btn"):
            score, wrong = 0, []
            for i, q in enumerate(mini_items):
                sel = mini_answers[i]
                ok = sel == q["answer"]; score += int(ok)
                if not ok:
                    wrong.append({
                        "문항": q["question"],
                        "선택한 답": sel if sel is not None else "(무응답)",
                        "정답": q["answer"],
                        "예문": q.get("ex","")
                    })
            st.success(f"점수: {score} / {len(mini_items)}")
            # 오답노트에 누적
            st.session_state["wrong_items"] = st.session_state.get("wrong_items", []) + wrong

    # 오답 복습 (간단 Leitner)
    with st.expander("🍊 오답 복습 🍒"):
        boxes = st.session_state.study["leitner"]
        st.write({f"박스 {k}": len(v) for k, v in boxes.items()})
        # 간단: 박스1 → 2 → 3 순
        pool = boxes["1"] or boxes["2"] or boxes["3"]
        if not pool:
            st.info("복습할 카드가 없어요. 퀴즈/학습에서 틀린 항목이 생기면 여기에 쌓입니다.")
        else:
            src, i = pool[0]  # ("lex", index) 형태
            if src == "lex" and not VOCAB.empty:
                r = VOCAB.iloc[i]
                st.markdown(f"**〔{r.get('유형','어휘')}〕 {r.get('어휘','')}**")
                if st.button("정답 (상위 박스로 이동)", key=f"leit_up_{i}"):
                    if pool is boxes["1"]:
                        boxes["2"].append(pool.pop(0))
                    elif pool is boxes["2"]:
                        boxes["3"].append(pool.pop(0))
                    else:
                        pool.pop(0)
                if st.button("오답 (1단계로)", key=f"leit_reset_{i}"):
                    if pool:
                        pair = pool.pop(0)
                    boxes["1"].append(("lex", i))

# ─────────────────────────────────────────────────────────────
# 오답노트 탭
# ─────────────────────────────────────────────────────────────
with tab_wrong:
    st.markdown("📘 **오답노트** (틀린 문제를 이곳에서 복습해보아요!)")

    # 비우기 버튼 (선택)
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("🧹 오답노트 비우기", key="clear_wrong_btn"):
            st.session_state["wrong_items"] = []
            st.success("오답노트를 비웠어요!")

    wrong_items = st.session_state.get("wrong_items", [])

    if not wrong_items:
        st.info("아직 오답이 없습니다! 퀴즈를 먼저 풀어보세요 😎")
    else:
        for i, w in enumerate(wrong_items, start=1):
            st.markdown(f"**Q{i}. {w.get('문항','(문항 정보 없음)')}**")
            st.write(f"- 선택한 답: {w.get('선택한 답', '(무응답)')}")
            st.write(f"- 정답: {w.get('정답', '-')}")
            ex = w.get("예문", "")
            if isinstance(ex, str) and ex.strip():
                st.write(f"- 예문: {ex}")
            st.divider()

# ─────────────────────────────────────────────────────────────
# 사이드바: 상태/확장 안내
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 상태")
    st.write(f"- 어휘 사전 로드: {'✅' if not VOCAB.empty else '❌'}")
    st.write(f"- 규정 카드 로드: {'✅' if RULES else '❌'}")
    # 기존: st.write(f"- 다의어 카드 로드: {'✅' if not POLY.empty else '❌'}")
    poly_ok = isinstance(POLY, pd.DataFrame) and not POLY.empty
    st.write(f"- 다의어 카드 로드: {'✅' if poly_ok else '❌'}")
    st.write(f"- 업로드 자료 색인: {'✅' if retriever is not None else '❌'}")
    st.divider()
    st.markdown("### 사용법")
    st.markdown("- 어휘: `교각살우 뜻`, `을씨년스럽다 의미`")
    st.markdown("- 규정: `같이 띄어쓰기`, `값이 발음`, `피자 표기`")
    st.markdown("- 다의어: `들다 다의어`, `달다 여러 뜻`, `치르다 뜻들`")
    st.markdown("- 퀴즈: 탭에서 **새 퀴즈 출제 → 제출**")
    st.markdown("- 업로드 RAG: 파일 올리고 자유 질의")







