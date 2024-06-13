import json
from typing import List

from dotenv import load_dotenv
import os

from huggingface_hub import HfApi
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from datasets import load_dataset


def load_env():
    load_dotenv('.env')

    os.getenv("LANGCHAIN_TRACING_V2")
    os.getenv("LANGCHAIN_ENDPOINT")
    os.getenv("LANGCHAIN_API_KEY")

    os.getenv("OPENAI_API_KEY")

    os.getenv("LM_URL")
    os.getenv("LM_LOCAL_URL")


def docs_load() -> List[str]:
    """
    문서를 읽는 함수
    """

    try:
        loader = TextLoader("corpus/정시 모집요강(동의대) 전처리 결과.txt", encoding="utf-8").load()
        return loader
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return []
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return []


def c_text_split(corpus: List[str]) -> List[str]:
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

    text_documents = c_text_splitter.split_documents(corpus)

    return text_documents


# 원하는 데이터 구조를 정의합니다.
class QAPair(BaseModel):
    question: str = Field(description="Question generated from the text")
    answer: str = Field(description="Answer related to the question")


def generations(text_documents):
    prompt = PromptTemplate.from_template(
        """Context information is below. You are only aware of this context and nothing else.
    ---------------------

    {context}

    ---------------------
    Given this context, generate only questions based on the below query.
    You are an Teacher/Professor in {domain}. 
    Your task is to provide exactly **{num_questions}** question(s) for an upcoming quiz/examination. 
    You are not to provide more or less than this number of questions. 
    The question(s) should be diverse in nature across the document. 
    The purpose of question(s) is to test the understanding of the students on the context information provided.
    You must also provide the answer to each question. The answer should be based on the context information provided only.

    Restrict the question(s) to the context information provided only.
    QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
    ANSWER should be a complete sentence.

    #Format:
    ```json
    {{
        "QUESTION": "국어국문학과에서 가군 일반학생 전형으로 몇 명을 모집하나요?",
        "ANSWER": "인문사회과학대학의 국어국문학과에서 가군 일반학생 전형으로 17명을 모집합니다."
    }},
    {{
        "QUESTION": "2024학년도 동의대의 정시 다군 전형에서 수능 최저기준이 적용되나요?",
        "ANSWER": "동의대학교 정시 전형은 모두 수능 최저기준이 적용되지 않습니다. 따라서, 전형요소 반영비율은 100% 수능 성적입니다. 추가적인 정보는 "https://ipsi.deu.ac.kr/main.do"에서 확인할 수 있습니다."
    }},
    {{
        "QUESTION": "나는 특수목적고등학교 학생인데, 동의대 학생부교과 농어촌학생전형으로 지원 가능한가요?",
        "ANSWER": "아니요, 특수목적고등학교 출신자는 동의대 학생부교과 농어촌학생전형으로 지원할 수 없습니다. 지원 자격에 따르면, 농어촌지역 또는 도서·벽지에 소재한 특수목적고 중 과학고, 국제고, 외국어고, 체육고, 예술고 출신자는 지원할 수 없습니다."    
    }},
    {{
        "QUESTION": "2024학년도 소프트웨어공학부 합격자 평균 성적을 알려줘.",
        "ANSWER": "### 2024학년도 소프트웨어공학부 합격자 평균 성적

#### 최초합격자
- **국어 + 수학 + 탐구(2개) (가산점 포함) 표준 변환 점수 합**: 330.11
- **영어 등급**: 3.30
- **수능 4개 영역 등급**: 4.12
- **수능 4개 영역 (가산점 포함) 표준 변환 점수 합**: 450.71

#### 최종등록자
- **국어 + 수학 + 탐구(2개) (가산점 포함) 표준 변환 점수 합**: 317.87
- **영어 등급**: 3.77
- **수능 4개 영역 등급**: 4.49
- **수능 4개 영역 (가산점 포함) 표준 변환 점수 합**: 434.70"    
    }},    
    {{
        "QUESTION": "e비즈니스학전공 예비38번 이정도면 예비합격 가능할지 궁금합니다.",
        "ANSWER": "e비즈니스학과의 최종등록자 데이터를 기준으로 보면, 충원합격(후보 순위) 31번까지 충원합격 된 것으로 나온다. 따라서, 현재 예비 38번이라면 e비즈니스학과에 예비 합격할 가능성이 낮다.

자세한 사항은 "https://ipsi.deu.ac.kr/main.do"에서 확인하시기 바랍니다."    
    }},    
    {{
        "QUESTION": "간호학과 정시 성적 2.52 추가 합격으로도 합격 안되겠죠?",
        "ANSWER": "간호학과의 최종등록자 데이터를 보면, 평균 영어 등급은 2.34이고, 표준 편차는 0.69입니다. 따라서, 2.52 등급은 평균보다 약간 낮은 수준입니다. 

추가 합격 여부는 지원자의 전체 성적, 경쟁률, 다른 지원자들의 성적 등에 따라 달라질 수 있습니다. 하지만, 평균보다 약간 낮은 성적이므로 합격 가능성이 낮을 수 있습니다. 

정확한 합격 여부는 학교의 입학처에 문의하는 것이 좋습니다. 추가 정보를 원하시면 "https://ipsi.deu.ac.kr/main.do"를 방문해 보세요."    
    }}
    ```
    """
    )

    # 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
    parser = JsonOutputParser(pydantic_object=QAPair)

    chain = (
            prompt
            | ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
            | parser
    )  # 체인을 구성합니다.

    qa_pair = []

    print(len(text_documents))

    # check = 0
    for text in text_documents:
        # if check > 4:
        #     break
        if text.page_content:
            qa_pair.append(
                chain.invoke(
                    {"context": text.page_content, "domain": "AI", "num_questions": "4"}
                )
            )
        # check += 1

    print("-----------------------------")
    for qa in qa_pair:
        print(qa)

    # 디버깅을 위한 데이터셋 추가
    additional_qa = [
        {
            "QUESTION": "너는 어느 대학교 입시 정보를 알려주는 챗봇 또는 질의응답 시스템이야??",
            "ANSWER": "저는 동의대학교의 정시 모집요강에 대한 질문에 답변하는 질의응답 시스템입니다. 동의대학교의 입시 정보를 알고 싶다면 저에게 물어보세요.",
        },
        {
            "QUESTION": "동의대학교 입시 정보 및 결과에 대해서 알려줘.",
            "ANSWER": "동의대학교 입시 관련 자세한 정보는 https://ipsi.deu.ac.kr/main.do 에서 확인할 수 있습니다. 입시 결과 및 모집요강에 대한 정보를 원한다면 해당 사이트를 방문해보는 것을 추천드립니다.",
        },
        {
            "QUESTION": "이 질의응답 시스템을 제작한 사람은 누구니?",
            "ANSWER": "동의대학교 컴퓨터소프트웨어공학과 석사과정 정민수입니다. 자세한 정보는 jminsu084@gmail.com으로 문의해주세요.",
        },
    ]

    qa_pair.extend(additional_qa)

    return qa_pair


def save_file(qa_pair_list):
    with open("test_automation/qa_pair_v2.jsonl", "w", encoding="utf-8") as f:
        for qa in qa_pair_list:
            # qa 형식이 딕셔너리인 경우
            if isinstance(qa, dict):
                qa_modified = {
                    "instruction": qa["QUESTION"],
                    "input": "",
                    "output": qa["ANSWER"],
                }
            # qa 형식이 list인 경우
            elif isinstance(qa, list):
                for q_a in qa:
                    qa_modified = {
                        "instruction": q_a["QUESTION"],
                        "input": "",
                        "output": q_a["ANSWER"],
                    }

            f.write(json.dumps(qa_modified, ensure_ascii=False) + "\n")


def huggingface_upload():
    # JSONL 파일 경로
    file_path = "test_automation/qa_pair_v2.jsonl"

    # JSONL 파일을 Dataset으로 로드
    dataset = load_dataset("json", data_files=file_path)

    print(dataset)

    # hfApi 인스턴스 생성
    api = HfApi()

    # 데이터셋을 업로드할 리포지토리 이름
    repo_name = "MinsuKorea/qna_pair_dataset"

    # 데이터셋을 허브에 푸시
    dataset.push_to_hub(repo_name, token=os.getenv("HUGGINGFACE_API_KEY"))

    print("업로드 완료")


def run():
    # 환경변수를 로드합니다.
    load_env()

    # 데이터를 읽어옵니다.
    corpus = docs_load()

    # 데이터를 분할합니다.
    text_documents = c_text_split(corpus)

    # 분할된 데이터를 출력합니다.
    for text_document in text_documents:
        print(text_document)

    # 질문 생성
    # qa_pair_list = generations(text_documents)

    # 질의응답 저장
    # save_file(qa_pair_list)

    # 허깅페이스 업로드
    huggingface_upload()


if __name__ == "__main__":
    run()
