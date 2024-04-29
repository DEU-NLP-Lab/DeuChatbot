from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import os
import shutil


def chat_llm():
    """
    채팅에 사용되는 거대언어모델 생성 함수
    :return: 답변해주는 거대언어모델
    """

    model_check = '2'

    while True:
        model_check = input(
            "채팅에 사용할 모델을 고르시오. 고르지 않을 경우 Google Gemini-1.5 Pro 모델을 기본으로 사용합니다.\n1: GPT-3.5\n2: "
            "Google Gemini-Pro\n3: EEVE Korean\n\n 선택 번호 : ")

        if model_check in ['1', '2', '3']:
            break
        else:
            print("잘못된 입력입니다. 1, 2, 3 중 하나를 선택해주세요.\n")

    if model_check == "1":
        os.environ['OPENAI_API_KEY'] = "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"  # 테스트 버전일 때

        # Retriever 적용
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0
        )
    elif model_check == "2":
        os.environ['GOOGLE_API_KEY'] = "AIzaSyBZuxIG0vS-XGSm6HDyrOaxbyRayY8yXDc"  # 테스트 버전일 때

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0
        )
    elif model_check == "3":
        # llm = ChatOllama(model="EEVE-Korean-10.8B:latest")
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
            # model="teddylee777/llama-3-8b-it-ko-chang-gguf",
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        # llm = ChatOllama(model="Llama-3:latest")

    return llm


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def db_qna(llm, db, query,):
    """
    벡터저장소에서 질문을 검색해서 적절한 답변을 찾아서 답하도록 하는 함수
    :param llm: 거대 언어 모델
    :param db: 벡터스토어
    :param query: 사용자 질문
    :return: 거대언어모델(LLM) 응답 결과
    """

    docs = db.similarity_search_with_relevance_scores(query, k=3, )

    for doc in docs:
        print("가장 유사한 문서:\n\n {}\n\n".format(doc[0].page_content))
        print("문서 유사도:\n {}".format(doc[1]))
        print("\n-------------------------")

    db = db.as_retriever(
        # search_type="mmr",
        # search_kwargs={'k': 3, 'fetch_k': 10},
        search_type='similarity_score_threshold',
        search_kwargs={'k': 3, 'score_threshold': 0.5},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert AI on a question and answer task.
                Use the "Following Context" when answering the question. If you don't know the answer, reply to the "Following Text" in the header and answer to the best of your knowledge, or if you do know the answer, answer without the "Following Text". If a question is asked in Korean, translate it to English and always answer in Korean.
                Following Text: "주어진 정보에서 답변을 찾지는 못했지만, 제가 아는 선에서 답을 말씀드려볼게요! **틀릴 수도 있으니 교차검증은 필수입니다!**"
                If the context is empty or you don't know the answer, tell them to contact "https://ipsi.deu.ac.kr/main.do".

                Following Context: {context}
                """,
            ),
            ("human", "Question: {question}"),
        ]
    )

    # template = """
    #         {context}
    #
    #         Question: {question}
    #         """
    #
    # prompt = ChatPromptTemplate.from_template(template)

    # print(f"--------------{db}--------------")

    chain = {
                "context": db | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } | prompt | llm

    response = chain.invoke(query)

    if not isinstance(llm, ChatOpenAI):
        print("\n\n{}".format(response.content))


def document_embedding(docs, model, save_directory):
    """
    OpenAI Embedding 모델을 사용하여 문서 임베딩하여 Chroma 벡터저장소(VectorStore)에 저장하는 함수
    :param model: 임베딩 모델 종류
    :param save_directory: 벡터저장소 저장 경로
    :param docs: 분할된 문서
    :return:
    """

    print("\n잠시만 기다려주세요.\n\n")

    # 벡터저장소가 이미 존재하는지 확인
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
        print(f"디렉토리 {save_directory}가 삭제되었습니다.\n")

    if isinstance(model, OpenAIEmbeddings):
        # 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
        # os.environ['OPENAI_API_KEY'] = input('발급 받은 OpenAI API Key를 입력해주세요: ')
        os.environ['OPENAI_API_KEY'] = "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"  # 테스트 버전일 때

    print("문서 벡터화를 시작합니다. ")
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    print("새로운 Chroma 데이터베이스가 생성되었습니다.\n")

    return db


def c_text_split(corpus):
    """
    CharacterTextSplitter를 사용하여 문서를 분할하도록 하는 함수
    :param corpus: 전처리 완료된 말뭉치
    :return: 분리된 청크
    """

    c_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="---",
        chunk_size=1500,
        chunk_overlap=0,
    )

    texts = c_text_splitter.split_text(corpus)

    text_documents = c_text_splitter.create_documents(texts)  # document로 만들기

    return text_documents


def docs_load():
    """
    문서 읽는 함수
    """

    with open("corpus/정시 모집요강(동의대) 전처리 결과.txt", "r", encoding="utf-8") as file:
        loader = file.read()

    return loader


def run():
    """
    챗봇 시작
    Document Load -> Text Splitter -> Ducument Embedding -> VectorStore save -> QA
    """

    # 문서 업로드
    loader = docs_load()

    # 문서 분할
    chunk = c_text_split(loader)

    # 문서 임베딩 및 벡터스토어 저장
    embedding_model_number = input(
        "문서 임베딩에 사용할 임베딩 모델을 고르시오. 고르지 않을 경우 HuggingFaceEmbeddings 모델을 기본으로 사용합니다.\n"
        "1: OpenAIEmbeddings()\n"
        "2: HuggingFaceEmbeddings()\n\n "
        "선택 번호 : ")

    if embedding_model_number == 1:
        model = OpenAIEmbeddings()
    else:
        model_name = "jhgan/ko-sroberta-multitask"  # 한국어 모델
        model_kwargs = {'device': 'cpu'}  # cpu를 사용하기 위해 설정
        encode_kwargs = {'normalize_embeddings': True}
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    db = document_embedding(chunk, model, save_directory="./chroma_db")

    # 채팅에 사용할 거대언어모델(LLM) 선택
    llm = chat_llm()

    # 정보 검색
    db = Chroma(persist_directory="./chroma_db", embedding_function=model)
    check = 'Y'  # 0이면 질문 가능
    while check == 'Y' or check == 'y':
        query = input("질문을 입력하세요 : ")

        db_qna(llm, db, query)

        check = input("\n\nY: 계속 질문한다.\nN: 프로그램 종료\n입력: ")


if __name__ == "__main__":
    os.environ['LANGCHAIN_TRACING_V2'] = "true"
    os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
    os.environ['LANGCHAIN_API_KEY'] = "ls__4de168fde3cc41e69287c27236fcc1ea"
    run()
