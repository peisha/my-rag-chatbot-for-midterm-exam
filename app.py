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

# 제목과 안내문 사이 2줄 공백
st.markdown("<br><br>", unsafe_allow_html=True)

# 안내문 (두 줄로 줄바꿈 적용)
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:14px; margin:10px 0 24px;'>"
    "규정 근거는 국립국어원의 『한글맞춤법·표준발음법·외래어·로마자 표기법』및 "
    "『표준국어대사전』임을 알려드립니다."
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
    ...
# ← 여기도 교체
VOCAB = load_lexicon_df()
RULES = load_rules_list() or []
POLY  = load_poly_df()

# ─────────────────────────────────────────────────────────────
# 파일 업로드 → 텍스트 추출 → 벡터DB 구성 (업로드시만)
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<h5 style='margin-bottom: 0.3em;'>🧐 별도로 학습시키고 싶은 자료가 있으신가요? 이곳에 업로드하세요! 👇📂</h5>",
    unsafe_allow_html=True
)
uploaded = st.file_uploader("txt 또는 pdf 파일을 업로드하세요", type=["txt", "pdf"])

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


def answer_rule(q: str) -> str:
    """rules.json에서 항목 키워드 부분일치 검색 (여러 개면 첫 항목)"""
    if not RULES:
        return "규정 데이터(rules.json)가 아직 없습니다. 먼저 data/rules.json을 채워 주세요."
    candidates = []
    for item in RULES:
        text_blob = f"{item.get('항목','')} {item.get('설명','')}"
        if any(tok in text_blob for tok in q.split()):
            candidates.append(item)
    target = candidates[0] if candidates else (RULES[0] if RULES else None)
    if not target:
        return "해당 규정을 찾지 못했어요. 질문을 조금만 다르게 써볼까요?"

    lines = [
        f"〔{target.get('규정명','규정')}〕 {target.get('항목','')}",
        target.get('설명',''),
    ]
    ex = target.get('예시','')
    if isinstance(ex, str) and ex.strip():
        lines.append(f"예시: {ex}")
    return "\n".join(lines)

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
# 탭 UI: 질문하기 | 퀴즈
# ─────────────────────────────────────────────────────────────
tab_ask, tab_quiz = st.tabs(["🧐 질문하기", "🤗 퀴즈 풀기"])

with tab_ask:
    # 라벨 부분을 HTML로 직접 출력 (엔터 포함)
    st.markdown(
        "<p style='font-size:18px; font-weight:600;'>🏄 저에게 질문을 해주세용 🐲!</p>"
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

# ─────────────────────────────────────────────────────────────
# 퀴즈 탭 (확장 버전: 제출 → 결과 → 새 퀴즈 버튼 순서)
# ─────────────────────────────────────────────────────────────
with tab_quiz:
    st.markdown("❤️쨔란! **랜덤 종합 퀴즈**를 풀어보세요!😘")

    # 처음 로드 시 data/ 모든 CSV 기반으로 혼합 퀴즈 생성
    if "quiz_items" not in st.session_state or not st.session_state.quiz_items:
        st.session_state.quiz_items = build_all_quiz_items(total=10)  # 총 문항 수 조절 가능
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = 0

    if not st.session_state.quiz_items:
        st.info("퀴즈를 만들 데이터가 부족합니다. data/ 폴더의 CSV들을 확인해 주세요.")
    else:
        answers = {}
        for i, item in enumerate(st.session_state.quiz_items):
            st.markdown(
                f"**Q{i+1}. {item['question']}**  \n<small>{item['meta']}</small>",
                unsafe_allow_html=True
            )
            key = f"quiz_q_{i}"
            choice = st.radio("보기", options=item["choices"], index=None, key=key, label_visibility="collapsed")
            answers[i] = choice
            st.divider()
        # 제출/채점/해설 부분은 기존 코드 유지


        # ✅ 제출 버튼 (맨 아래)
        if st.button("✅ 제출", type="primary", use_container_width=True):
            score = 0
            results = []
            for i, item in enumerate(st.session_state.quiz_items):
                sel = answers.get(i)
                ok = (sel == item["answer"])
                score += int(ok)
                results.append((ok, sel, item))
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True

            st.success(f"점수: **{score} / {len(st.session_state.quiz_items)}**")
            with st.expander("정답 및 해설 보기"):
                for i, (ok, sel, item) in enumerate(results, start=1):
                    icon = "✅" if ok else "❌"
                    sel_txt = sel if sel is not None else "(무응답)"
                    st.markdown(f"**{icon} Q{i}. {item['question']}**")
                    st.write(f"- 선택: {sel_txt}")
                    st.write(f"- 정답: {item['answer']}")
                    if item["ex"]:
                        st.write(f"- 예문: {item['ex']}")
                    st.write("---")

        # 🔄 새 퀴즈 출제 버튼 (제출 아래)
        if st.button("🔄 새 퀴즈 출제", use_container_width=True):
           st.session_state.quiz_items = build_all_quiz_items(total=10)
           st.session_state.quiz_submitted = False
           st.session_state.quiz_score = 0
           st.rerun()

                    
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






































