import tiktoken  # openAI 에서 제공하는 오픈 소스 토크나이저
from langchain.text_splitter import CharacterTextSplitter
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


def db_qna(q):
    """
    벡터저장소에서 질문을 검색해서 적절한 답변을 찾아서 답하도록 하는 함수
    :param q: 사용자 질문
    :return: 답변
    """
    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    docs = db.similarity_search(q)

    # test = db.similarity_search_with_relevance_scores(query, k=5)  # 유사도가 높은 문서 몇개까지 반환받을 건지 설정할 때 k 매개변수 사용 / 유사도 높은 상위 3개 문서를 뽑아서 docs에 반환
    # for doc in test:
    #     print("가장 유사한 문서:\n\n {}\n\n".format(doc[0].page_content))
    #     print("문서 유사도:\n {}".format(doc[1]))
    #     print("\n-------------------------")


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

    result = qa(q)
    # print("\n\n{}".format(result))
    # print("\n\n{}".format(result['source_documents'][0].page_content))  # 인덱스 맨 마지막 것이 가장 유사도가 높은거 같음 => 아래 실행해보니 아님. 후보군 3개 중에서 짜집기해서 답 내는 것 같음


def use_openai_embedding(docs, save_directory):
    """
    openai embedding model을 사용하여 벡터저장소 (vectorStore)에 저장하려고 할 때 사용하는 함수
    :param docs: 분할된 문서
    :param save_directory: 저장 경로
    :return: 
    """

    # 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
    os.environ['OPENAI_API_KEY'] = input('발급 받은 OpenAI API Key를 입력해주세요: ')

    # 벡터저장소가 이미 존재하는지 확인
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
        print(f"디렉토리 {save_directory}가 삭제되었습니다.\n")

    print("문서 벡터화를 시작합니다. ")
    Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory=save_directory)

    print("새로운 Chroma 데이터베이스가 생성되었습니다.")


def rc_text_split(corpus):
    """
    RecursiveCharacterTextSplitter를 사용하여 문서를 분할하도록 하는 함수
    :param corpus:
    :return:
    """
    rc_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=tiktoken_len,  # 단위를 토큰으로 함
    )

    texts = rc_text_splitter.split_text(loader)
    # print(texts[0])

    text_documents = rc_text_splitter.create_documents(texts)  # document로 만들기

    # print(f"글자 수 {len(text_documents[0].page_content)}")
    # print(f"토큰 수 {tiktoken_len(text_documents[0].page_content)}")

    # token 사이즈 출력

    token_list = []
    for i in range(len(text_documents)):
        token_list.append(tiktoken_len(text_documents[i].page_content))

    # print(token_list)
    # print(len(token_list))


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

    texts = c_text_splitter.split_text(loader)
    # print(texts[0])

    text_documents = c_text_splitter.create_documents(texts)  # document로 만들기

    # print(f"글자 수 {len(text_documents[0].page_content)}")
    # print(f"토큰 수 {tiktoken_len(text_documents[0].page_content)}")

    # token 사이즈 출력

    token_list = []
    for i in range(len(text_documents)):
        token_list.append(tiktoken_len(text_documents[i].page_content))

    # print(token_list)
    # print(len(token_list))

    # 문서 임베딩 작업 결과 벡터저장소에 저장
    # use_openai_embedding(text_documents, save_directory="./chroma_db")


if __name__ == "__main__":
    with open("corpus/정시 모집요강(동의대) 전처리 결과.txt", "r", encoding="utf-8") as f:
        # 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
        # os.environ['OPENAI_API_KEY'] = input('발급 받은 OpenAI API Key를 입력해주세요: ')  # 정규 버전일 때
        os.environ['OPENAI_API_KEY'] = "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"  # 테스트 버전일 때

        loader = f.read()
        
        # 문서 임베딩 작업할 때 사용할 splitter에 맞추어 주석 풀기
        # c_text_split(loader)  # CharacterTextSplitter 사용할 때
        # rc_text_split(loader)  # RecursiveCharacterTextSplitter 사용할 때

        # 이 부분만 수정해서 사용하면 됨
        # query = "원서 접수 날짜는 언제야?"
        query = "디자인조형학과의 실기종목과 준비물, 실기주에, 화지크기, 시간에 대해서 알려줘"
        db_qna(query)
