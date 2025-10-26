import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) .env 로드 (Downloads 폴더의 .env)
load_dotenv(override=True)

# 2) Streamlit 기본 설정
st.set_page_config(page_title="My RAG Chatbot")
st.title("📘 파일 업로드형 RAG 챗봇")
st.caption("txt 또는 pdf 파일을 업로드하세요")

# 3) 파일 업로더
uploaded = st.file_uploader("파일을 선택하세요", type=["txt", "pdf"])

# 4) 파일 텍스트 로딩 함수
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

# 5) 업로드 파일 처리
if uploaded:
    text = load_text(uploaded)

    # (a) 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    # (b) 임베딩: 비용 없는 로컬 임베딩 사용
    #  - 모델 다운로드는 최초 1회만 이루어지고, 이후 캐시를 사용합니다.
    with st.spinner("임베딩 생성 중… (최초 1회는 모델 다운로드로 오래 걸릴 수 있어요)"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = FAISS.from_documents(docs, embeddings)

    # (c) 리트리버
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # (d) LLM: OpenAI 사용 (응답 생성용)
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY가 설정되어 있지 않습니다. (.env 파일 확인)")
        st.stop()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # (e) 프롬프트
    prompt = ChatPromptTemplate.from_template(
        "다음 컨텍스트를 참고해서 한국어로 간결하고 정확히 답변하세요.\n"
        "필요하면 근거를 요약해서 덧붙이세요.\n\n"
        "컨텍스트:\n{context}\n\n"
        "질문: {question}"
    )

    # (f) RAG 체인 실행 함수
    def run_rag(question: str) -> str:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    # (g) 질문 입력 UI
    query = st.text_input("질문을 입력하세요:")
    if query:
        with st.spinner("검색/생성 중…"):
            try:
                answer = run_rag(query)
                st.write("💬 답변:", answer)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
else:
    st.info("📂 상단에서 파일을 업로드하면 준비가 완료됩니다.")
