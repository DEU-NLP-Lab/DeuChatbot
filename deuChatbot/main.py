import tiktoken  # openAI 에서 제공하는 오픈 소스 토크나이저
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import shutil


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def chat_llm(db):
    """
    채팅에 사용되는 거대언어모델 생성 함수
    :param db: 벡터저장소
    :return: 답변해주는 거대언어모델
    """

    os.environ['OPENAI_API_KEY'] = "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"  # 테스트 버전일 때

    # Retriever 적용
    openai = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0
    )

    # chroma 객체를 retriever를 활용할 건데 mmr 즉, 최대한 다양하게 답변을 구성할 것이다. 근데 어떤식으로 답변을 구성할 것이냐면
    # 총 10개의 연관성있는 문서를 뽑은 다음에 최대한 다양하게 조합을 구성을 하되 세 개만 컨텍스트로 LLM에게 넘겨줘라는 것이다.
    qa = RetrievalQA.from_chain_type(
        llm=openai,
        chain_type="stuff",  # 전처리 데이터가 현재 청크 사이즈 문제가 있어 토큰 이슈가 발생할 가능성 있음
        retriever=db.as_retriever(
            # docsearch에서 chroma를 저장소로 사용하는 것이 아니라 as_retriever 연관성 높은 벡터를 찾는 검색기로 사용하기 위해서 as_retriever 함수 사용
            search_type="mmr",
            # 벡터 저장소에서 사용자의 질문과 연관돼 있는 텍스트 청크를 뽑아오는 것인데, 거기서 연관성 높은 문서를 뽑아올 때 연관성 높은 풀 몇 개 중에서 최대한 다양하게 답변을 컨텍스트를
            # 조합해서 llm에게 컨텍스트로 던져줘라는 것이다. 여러 가지 소스를 다양한 소스를 참고해서 LLM 답변을 얻고 싶을 때 서치타입 mmr을 선언하면 됨
            search_kwargs={'k': 3, 'fetch_k': 10}
            # mmr을 어떤식으로 구현할지에 대해서 구체적으로 명시해주는 부분 / 연관 있는 문서 후보군을 fetch_k개 만큼 만들고 LLM에게 최종적으로 넘길 때 k개만큼 넘기겠다
        ),
        return_source_documents=True
    )

    return qa


def db_qna(db, q):
    """
    벡터저장소에서 질문을 검색해서 적절한 답변을 찾아서 답하도록 하는 함수
    :param db: 벡터스토어
    :param q: 사용자 질문
    :return: 추가 질문 유무
    """

    qa = chat_llm(db)

    result = qa(q)
    # print("\n\n{}".format(result))
    # print("\n\n{}".format(result['source_documents'][0].page_content))  # 인덱스 맨 마지막 것이 가장 유사도가 높은거 같음 => 아래 실행해보니 아님. 후보군 3개 중에서 짜집기해서 답 내는 것 같음

    return input("\n\nY: 계속 질문한다.\nN: 프로그램 종료\n입력: ")


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
    Chroma.from_documents(docs, model, persist_directory=save_directory)
    print("새로운 Chroma 데이터베이스가 생성되었습니다.\n")


def c_text_split(corpus):
    """
    CharacterTextSplitter를 사용하여 문서를 분할하도록 하는 함수
    :param corpus: 전처리 완료된 말뭉치
    :return:
    """

    c_text_splitter = CharacterTextSplitter(
        separator="---",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=tiktoken_len,  # 단위를 토큰으로 함
    )

    texts = c_text_splitter.split_text(corpus)

    text_documents = c_text_splitter.create_documents(texts)  # document로 만들기

    # token 사이즈 출력

    # token_list = []
    # for i in range(len(text_documents)):
    #     token_list.append(tiktoken_len(text_documents[i].page_content))

    # print(token_list)

    return text_documents


def run():
    """
    챗봇 시작
    Document Load -> Text Splitter -> Ducument Embedding -> VectorStore save -> QA
    :return:
    """

    # 문서 업로드
    with open("corpus/정시 모집요강(동의대) 전처리 결과.txt", "r", encoding="utf-8") as file:
        loader = file.read()

        # 문서 분할
        chunk = c_text_split(loader)

        # 문서 임베딩 및 벡터스토어 저장
        embedding_model_number = input(
            "문서 임베딩에 사용할 임베딩 모델을 고르시오. 고르지 않을 경우 HuggingFaceEmbeddings 모델을 기본으로 사용합니다.\n1: OpenAIEmbeddings()\n2: HuggingFaceEmbeddings()\n\n 선택 번호 : ")

        if embedding_model_number == 1:
            model = OpenAIEmbeddings()
        else:
            model_name = "jhgan/ko-sbert-nli"  # 한국어 모델
            model_kwargs = {'device': 'cpu'}  # cpu를 사용하기 위해 설정
            encode_kwargs = {'normalize_embeddings': True}
            model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

        document_embedding(chunk, model, save_directory="./chroma_db")

        # 정보 검색
        db = Chroma(persist_directory="./chroma_db", embedding_function=model)
        check = 'Y'  # 0이면 질문 가능
        while check == 'Y' or check == 'y':
            query = input("질문을 입력하세요 : ")
            check = db_qna(db, query)


if __name__ == "__main__":
    run()
