from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

st.set_page_config(
    page_title="DeuChatbot",
    page_icon="https://deu.ac.kr/Upload/www/favicon/2018/1213091943440.ico"
)

st.title("동의대학교 정시 입시챗봇")


class ChatCallBackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


with st.sidebar:
    file = st.file_uploader(
        "입시 정보 파일을 추가해주세요.",
        type=["txt", "pdf"]
    )

    print(file)


def chat_llm(model):
    """
    채팅에 사용되는 거대언어모델 생성 함수
    :return: 답변해주는 거대언어모델
    """

    if model == "OpenAI - GPT-3.5":
        os.environ['OPENAI_API_KEY'] = "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"  # 테스트 버전일 때

        # Retriever 적용
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            streaming=True, callbacks=[ChatCallBackHandler()],
            temperature=0
        )

    return llm


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a Dong-Eui University admissions officer. Provide precise and courteous answers to the various users asking questions about college admissions. but Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
#             Before responding, think step by step do it.
#
#             Context: {context}
#             """,
#         ),
#         ("human", "{question}"),
#     ]
# )

model = st.selectbox(
    "사용할 거대언어모델(LLM)을 선택하시오.",
    (
        "OpenAI - GPT-3.5",
    )
)


@st.cache_data(show_spinner="Embedding file...")
def chunk_embedding(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    if not os.path.exists("./.cache/files"):
        os.mkdir("./.cache/files")

    if not os.path.exists("./.cache/embeddings"):
        os.mkdir("./.cache/embeddings")

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    c_text_splitter = None

    loader = None

    if file.type == "application/pdf":
        c_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n\n",
            chunk_size=1500,
            chunk_overlap=0,
        )

        loader = PyPDFLoader(file_path)
    elif file.type == "text/plain":
        c_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="---",
            chunk_size=1500,
            chunk_overlap=0,
        )
        loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=c_text_splitter)

    model_name = "jhgan/ko-sbert-nli"  # 한국어 모델
    model_kwargs = {'device': 'cpu'}  # cpu를 사용하기 위해 설정
    encode_kwargs = {'normalize_embeddings': True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embedding_model, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10},
    )

    return retriever


if file:
    llm = chat_llm(model)

    retriever = chunk_embedding(file)

    send_message("질문을 입력해주세요.", "ai", save=False)

    paint_history()

    message = st.chat_input("대학 입시 (정시)에 관해서 궁금한 것을 질문해주세요.")

    if message:
        send_message(message, "human")

        template = """
                    {context}
                    
                    Question: {question}
                    """

        prompt = ChatPromptTemplate.from_template(template)

        chain = {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                } | prompt | llm

        print(prompt)

        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
