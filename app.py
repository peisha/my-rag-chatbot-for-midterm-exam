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
st.title("✨😎 오로지 당신만을 위한~! KBS 한국어능력시험 쌤 💕💫")
st.caption("❗본 자료의 규정 근거는 국립국어원에서 기술한 『한글맞춤법/표준발음법/외래어·로마자 표기법』, 그리고 『표준국어대사전』 두 가지에 있음을 알려드립니다.😘")

# ─────────────────────────────────────────────────────────────
# 데이터 로더 (어휘/규정/다의어) - 업로드 없이도 동작
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_vocab_df():
    """data/vocab.csv (유형,표제어,뜻풀이,예문,비고)"""
    path = "data/vocab.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["유형","표제어","뜻풀이","예문","비고"])
    return pd.read_csv(path)
@st.cache_data
def load_rules_list():
    """data/rules.json (규정명,항목,설명,예시)"""
    path = "data/rules.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_poly_df():
    """data/polysemy.csv (표제어,의미번호,뜻,예문)"""
    path = "data/polysemy.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["표제어","의미번호","뜻","예문"])
    return pd.read_csv(path)

VOCAB = load_vocab_df()
RULES = load_rules_list()
POLY  = load_poly_df()

# ─────────────────────────────────────────────────────────────
# 파일 업로드 → 텍스트 추출 → 벡터DB 구성 (업로드시만)
# ─────────────────────────────────────────────────────────────
st.subheader("📂 학습 자료 업로드 (선택)")
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
        return "사전 데이터(vocab.csv)가 아직 없습니다. 먼저 data/vocab.csv를 채워 주세요."
    hit = VOCAB[VOCAB["표제어"].apply(lambda w: isinstance(w, str) and w in q)]
    if len(hit):
        row = hit.iloc[0]
        lines = [
            f"〔{row.get('유형','어휘')}〕 {row.get('표제어','-')}",
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
    return "사전에 직접 일치하는 표제어가 없어요.\n\n" + back

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
def build_quiz_items(df: pd.DataFrame, n: int = 3):
    """
    df: columns = [유형, 표제어, 뜻풀이, 예문, 비고]
    반환: [{"question","choices","answer","ex","meta"}, ...]
    """
    # 최소 컬럼 체크
    need_cols = {"유형", "표제어", "뜻풀이"}
    if not need_cols.issubset(df.columns):
        return []

    base = df.dropna(subset=["표제어", "뜻풀이"]).copy()
    if base.empty:
        return []

    items = []
    # 표본을 섞어서 위에서부터 채택
    base = base.sample(frac=1.0, random_state=None)

    for _, row in base.iterrows():
        q_word   = str(row["표제어"]).strip()
        correct  = str(row["뜻풀이"]).strip()
        cat      = str(row.get("유형", "어휘")).strip()
        example  = str(row.get("예문", "")).strip()

        if not q_word or not correct:
            continue

        # 같은 유형에서 오답 고르기 (부족하면 전체에서 보충)
        same_pool = base[(base["유형"] == cat) & (base["뜻풀이"] != correct)]["뜻풀이"].dropna().unique().tolist()
        if len(same_pool) < 3:
            same_pool = base[base["뜻풀이"] != correct]["뜻풀이"].dropna().unique().tolist()

        if len(same_pool) < 3:
            # 오답이 3개 미만이면 건너뛰기
            continue

        distractors = random.sample(same_pool, 3)
        choices = distractors + [correct]
        random.shuffle(choices)

        # ✅ 메타는 [유형]만
        meta = f"[{cat}]"

        items.append({
            "question": f"‘{q_word}’의 뜻으로 가장 알맞은 것은?",
            "choices": choices,
            "answer": correct,
            "ex": example,
            "meta": meta
        })

        if len(items) >= n:
            break

    return items

# ─────────────────────────────────────────────────────────────
# 탭 UI: 질문하기 | 퀴즈
# ─────────────────────────────────────────────────────────────
tab_ask, tab_quiz = st.tabs(["🧐 질문하기", "🤗 퀴즈 풀기"])

with tab_ask:
    user_q = st.text_input(
        "🙌 질문 예시: 교각살우의 뜻이 궁금해요, 늑막염의 표준 발음을 알려주세요, 자료에서 ~는 어디에 나오나요?"
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
    st.markdown("❤️짜란! **랜덤 퀴즈** 3문항을 풀어보세요!😘")

    if VOCAB.empty or len(VOCAB.dropna(subset=["표제어","뜻풀이"])) < 4:
        st.info("퀴즈를 만들려면 `data/vocab.csv`에 최소 4개 이상의 항목이 필요합니다.")
    else:
        # 초기 세션 상태
        if "quiz_items" not in st.session_state:
            st.session_state.quiz_items = build_quiz_items(VOCAB, n=3)
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0

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
                options=item["choices"],
                index=None,
                key=key,
                label_visibility="collapsed"
            )
            answers[i] = choice
            st.divider()

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
            st.session_state.quiz_items = build_quiz_items(VOCAB, n=3)
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.rerun()
                    
# ─────────────────────────────────────────────────────────────
# 사이드바: 상태/확장 안내
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 상태")
    st.write(f"- 어휘 사전 로드: {'✅' if not VOCAB.empty else '❌'}")
    st.write(f"- 규정 카드 로드: {'✅' if len(RULES)>0 else '❌'}")
    st.write(f"- 다의어 카드 로드: {'✅' if not POLY.empty else '❌'}")
    st.write(f"- 업로드 자료 색인: {'✅' if retriever is not None else '❌'}")
    st.divider()
    st.markdown("### 사용법")
    st.markdown("- 어휘: `교각살우 뜻`, `을씨년스럽다 의미`")
    st.markdown("- 규정: `같이 띄어쓰기`, `값이 발음`, `피자 표기`")
    st.markdown("- 다의어: `들다 다의어`, `달다 여러 뜻`, `치르다 뜻들`")
    st.markdown("- 퀴즈: 탭에서 **새 퀴즈 출제 → 제출**")
    st.markdown("- 업로드 RAG: 파일 올리고 자유 질의")



