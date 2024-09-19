from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
import os
import pandas as pd


class ChatBotSystem:
    def __init__(self):
        self.system = "ChatBotSystem"

    def load_env(self):
        load_dotenv('.env')

        os.getenv("LANGCHAIN_TRACING_V2")
        os.getenv("LANGCHAIN_ENDPOINT")
        os.getenv("LANGCHAIN_API_KEY")

        os.getenv("OPENAI_API_KEY")
        os.getenv("ANTHROPIC_API_KEY")
        os.getenv("GOOGLE_API_KEY")
        os.getenv("UPSTAGE_API_KEY")

        os.getenv("LM_URL")
        os.getenv("LM_LOCAL_URL")

        os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def chat_llm(self):
        """
        채팅에 사용되는 거대언어모델 생성 함수
        :return: 답변해주는 거대언어모델
        """
        load_dotenv('.env')

        while True:
            model_check = input(
                "채팅에 사용할 모델을 고르시오. 고르지 않을 경우 Google Gemini-1.5 Pro 모델을 기본으로 사용합니다.\n"
                "1: GPT-4o-mini\n2: GPT-4-turbo\n3: GPT-4o\n"
                "4: EEVE Korean\n5: Qwen1.5-14B-Chat\n6: Llama-3-MAAL-8B-Instruct-v0.1\n\n "
                "선택 번호 : ")

            if model_check in ['1', '2', '3', '4', '5', '6']:
                break
            else:
                print("잘못된 입력입니다. 1, 2, 3, 4, 5, 6중 하나를 선택해주세요.\n")

        model_info = self.get_model_info(model_check)
        model_class = model_info["model_class"]
        model_kwargs = {
            "temperature": 0,
        }

        if model_class == ChatOpenAI:
            model_kwargs.update({
                "model_name": model_info["model_name"],
                "streaming": True,
                "callbacks": [StreamingStdOutCallbackHandler()],
            })
            if "base_url" in model_info:
                model_kwargs["base_url"] = model_info["base_url"]
            if "api_key" in model_info:
                model_kwargs["api_key"] = model_info["api_key"]

        try:
            llm = model_class(**model_kwargs)
        except Exception as e:
            print(f"Error initializing the model: {str(e)}")
            return None, None

        return llm, model_check

    def get_model_info(self, model_check: str) -> dict:
        """
        선택된 모델에 알맞은 정보를 가공하는 함수
        """
        models = {
            "1": {"model_name": "gpt-4o-mini", "model_class": ChatOpenAI},
            "2": {"model_name": "gpt-4-turbo", "model_class": ChatOpenAI},
            "3": {"model_name": "gpt-4o", "model_class": ChatOpenAI},
            "4": {"model_name": "teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf", "model_class": ChatOpenAI,
                  "base_url": os.getenv("LM_URL"), "api_key": "lm-studio"},
            # "base_url": os.getenv("LM_LOCAL_URL"), "api_key": "lm-studio"},
            "5": {"model_name": "Qwen/Qwen1.5-14B-Chat-GGUF", "model_class": ChatOpenAI,
                  "base_url": os.getenv("LM_URL"),
                  "api_key": "lm-studio"},
            "6": {"model_name": "asiansoul/Llama-3-MAAL-8B-Instruct-v0.1-GGUF", "model_class": ChatOpenAI,
                  "base_url": os.getenv("LM_URL"), "api_key": "lm-studio"},
        }

        return models.get(model_check)

    def format_docs(self, docs):
        return "\n\n".join(document.page_content for document in docs)

    def qna(self, llm, correct_answer, answer):
        """
        GPTScore, G-Eval 논문에 따른 QnA 평가 함수

        :param llm: 거대 언어 모델
        :param correct_answer: 모범 답안
        :param answer: 거대언어모델이 생성한 답안
        :return:
        """

        # from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        # import torch
        #
        # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        # model = AutoModelForCausalLM.from_pretrained(
        #     "google/gemma-2-9b-it",
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        # )
        #
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        #
        # llm = HuggingFacePipeline(pipeline=pipe)

        # 파일 경로 설정
        file_path = os.path.join("prompt", "prompt.txt")

        # UTF-8 인코딩 방식으로 파일 읽어오기
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()

        prompt = PromptTemplate.from_template(template)

        gpt_score_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    # GPTScore Prompt Template

                    ## Factuality
                    Evaluate Factuality in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the factuality of the generated response by comparing it to the source text.
                    
                    **Evaluation Criteria**:
                    - Factuality (1-5): Does the generated response accurately reflect the factual statements found in the source text? A score of 1 means that the response contains multiple inaccuracies or fabricated information, while a score of 5 means that the response is entirely accurate and preserves all factual details from the source text.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to identify key factual statements and details.
                    2. Review the generated response and compare it to the source text, focusing on the accuracy and integrity of the facts presented.
                    3. Assign a score for factuality on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    ## Consistency
                    Evaluate Consistency in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the consistency of the generated response by comparing it to the source text.
                    
                    **Evaluation Criteria**:
                    - Consistency (1-5): Does the generated response consistently provide information that aligns with the source text? A score of 1 means that the response contains contradictory or conflicting information, while a score of 5 means that the response is fully consistent with no discrepancies.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand the key points and details.
                    2. Review the generated response and compare it to the source text, focusing on the consistency of the information provided.
                    3. Provide a detailed explanation of your evaluation, noting any inconsistencies or confirming the consistency of the response.
                    4. Assign a score for consistency on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the consistency, highlighting specific aspects where the generated response aligns or diverges from the source text.
                    
                    ## Relevance
                    Evaluate Relevance in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the relevance of the generated response by comparing it to the source text.
                    
                    **Evaluation Criteria**:
                    - Relevance (1-5): How well does the generated response relate to the source text? A score of 1 means that the response is largely irrelevant or off-topic, while a score of 5 means that the response is highly relevant and directly addresses the key points of the source text.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand its main topics and key points.
                    2. Review the generated response and compare it to the source text, focusing on how well it addresses the main topics and key points.
                    3. Provide a detailed explanation of your evaluation, highlighting specific areas where the generated response is relevant or irrelevant to the source text.
                    4. Assign a score for relevance on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the relevance, highlighting specific aspects where the generated response aligns or diverges from the main topics of the source text.
                    
                    ## Fluency
                    Evaluate Fluency in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the fluency of the generated response by assessing its grammatical correctness and readability.
                    
                    **Evaluation Criteria**:
                    - Fluency (1-5): How well is the generated response written in terms of grammar, syntax, and overall readability? A score of 1 means the response is poorly written, with numerous grammatical errors or awkward phrasing, while a score of 5 means the response is highly fluent, with excellent grammar and smooth readability.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand its style and tone.
                    2. Review the generated response, focusing on its grammatical correctness, sentence structure, and overall readability.
                    3. Provide a detailed explanation of your evaluation, noting any grammatical errors, awkward phrasing, or confirming the fluency of the response.
                    4. Assign a score for fluency on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the fluency, highlighting specific aspects where the generated response is grammatically correct, easy to read, or where it may have issues with fluency.
                    
                    ## Coherence
                    Evaluate Coherence in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the coherence of the generated response by comparing it to the source text.
                    
                    **Evaluation Criteria**:
                    - Coherence (1-5): How well does the generated response make sense as a unified piece of text? A score of 1 means that the response is disjointed or lacks logical flow, while a score of 5 means that the response is well-structured and logically consistent throughout.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand its overall structure and key points.
                    2. Review the generated response and assess its logical flow, structure, and overall coherence.
                    3. Provide a detailed explanation of your evaluation, noting any logical inconsistencies or confirming the coherence of the response.
                    4. Assign a score for coherence on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the coherence, highlighting specific aspects where the generated response aligns or diverges in its logical flow and structure compared to the source text.
                    
                    ## Accuracy
                    Evaluate Accuracy in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the accuracy of the generated response by comparing it to the source text.
                    
                    **Evaluation Criteria**:
                    - Accuracy (1-5): Are there any inaccuracies, omissions, or unfactual content in the generated response? A score of 1 means that the response contains significant inaccuracies or incorrect information, while a score of 5 means that the response is fully accurate and correctly reflects the source text.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to identify all the key facts and details.
                    2. Review the generated response and compare it to the source text, focusing on the accuracy of the information provided.
                    3. Provide a detailed explanation of your evaluation, highlighting any inaccuracies, omissions, or confirming the accuracy of the response.
                    4. Assign a score for accuracy on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the accuracy, highlighting specific aspects where the generated response aligns with or diverges from the facts in the source text.
                    
                    ## Multidimensional Quality
                    Evaluate Multidimensional Quality in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the overall quality of the generated response across multiple dimensions, including factuality, coherence, relevance, and accuracy.
                    
                    **Evaluation Criteria**:
                    - Multidimensional Quality (1-5): How well does the generated response perform across all relevant quality dimensions (factuality, coherence, relevance, accuracy)? A score of 1 means the response performs poorly in most or all dimensions, while a score of 5 means the response performs excellently across all dimensions.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand the main content and quality dimensions.
                    2. Review the generated response, assessing its performance in terms of factuality, coherence, relevance, and accuracy.
                    3. Provide a detailed explanation of your evaluation, highlighting the strengths and weaknesses of the response across all quality dimensions.
                    4. Assign a score for overall multidimensional quality on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the overall quality, considering the factuality, coherence, relevance, and accuracy of the generated response.
                    
                    ## Semantic Appropriateness
                    Evaluate Semantic Appropriateness in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the semantic appropriateness of the generated response by comparing it to the source text.
                    
                    **Evaluation Criteria**:
                    - Semantic Appropriateness (1-5): How well does the generated response convey meaning that is appropriate and aligned with the context of the source text? A score of 1 means the response is semantically inappropriate or off-context, while a score of 5 means the response is fully appropriate and semantically consistent with the source text.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand its meaning and context.
                    2. Review the generated response and assess whether the meaning it conveys is appropriate and aligned with the source text.
                    3. Provide a detailed explanation of your evaluation, highlighting areas where the generated response is semantically appropriate or where it diverges from the intended meaning.
                    4. Assign a score for semantic appropriateness on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the semantic appropriateness, highlighting specific aspects where the generated response aligns or diverges from the meaning and context of the source text.
                    
                    ## Understandability
                    Evaluate Understandability in the Generated Response
                    
                    You will be given a source text and a generated response. Your task is to evaluate the understandability of the generated response by considering how clear and easy it is to comprehend.
                    
                    **Evaluation Criteria**:
                    - Understandability (1-5): How easy is it to understand the generated response? A score of 1 means that the response is confusing, unclear, or difficult to comprehend, while a score of 5 means that the response is very clear, straightforward, and easy to understand.
                    
                    **Evaluation Steps**:
                    1. Carefully read the source text to understand its main points and context.
                    2. Review the generated response, focusing on how clearly the information is presented and whether it is easy to follow.
                    3. Provide a detailed explanation of your evaluation, noting any areas where the response is particularly clear or where it may be confusing.
                    4. Assign a score for understandability on a scale of 1 to 5 based on the Evaluation Criteria.
                    
                    **Important**: Ensure that all responses, including explanations and scores, are generated in Korean.
                    
                    Evaluation Explanation:
                    - Provide an analysis of the understandability, highlighting specific aspects where the generated response is clear or where it may be difficult to comprehend.
                    
                    ## Output Score
                    **Example**:
                    Source Text:
                    {source_text}
                    
                    Generated Response:
                    {generated_response}
                    
                    1. Factuality Score (1-5):
                    2. Consistency Score (1-5):
                    3. Relevance Score (1-5):
                    4. Fluency Score (1-5):
                    5. Coherence Score (1-5):
                    6. Accuracy Score (1-5):
                    7. Multidimensional Quality Score (1-5):
                    8. Semantic Appropriateness Score (1-5):
                    9. Understandability Score (1-5):
                    
                    {{
                        "Factuality Score": 4,
                        "Consistency Score": 3,
                        "Relevance Score": 5,
                        "Fluency Score": 4,
                        "Coherence Score": 4,
                        "Accuracy Score": 4,
                        "Multidimensional Quality Score": 4,
                        "Semantic Appropriateness Score": 4,
                        "Understandability Score": 4
                    }}                    

                    """,
                ),
                ("human", "generated_response: {generated_response}"),
            ]
        )

        gpt_score_chain = {
                    "source_text": lambda x: correct_answer,
                    "generated_response": RunnablePassthrough()
                } | gpt_score_prompt | llm | StrOutputParser()

        gpt_score_response = gpt_score_chain.invoke({"source_text": correct_answer, "generated_response": answer})

        if not isinstance(llm, ChatOpenAI):
            print("\n\n{}".format(gpt_score_response))

        return gpt_score_response

    def file_read(self):
        """
        엑셀 파일을 읽어서 데이터프레임으로 반환하는 함수
        :return: 파일 내용
        """

        file_path = "research_result/"

        # 전체 파일 경로 리스트를 선언
        full_file_paths = []

        # 경로가 디렉토리인지 확인
        if os.path.isdir(file_path):
            # 해당 디렉토리의 파일 목록을 가져옴
            file_list = os.listdir(file_path)

            # 파일 경로를 결합하여 리스트로 추가
            full_file_paths = [os.path.join(file_path, file) for file in file_list]

        # if 블록 바깥에서도 full_file_paths 사용 가능
        print(full_file_paths)





        return full_file_paths


# def auto_question():
#     """
#
#     """
#
#     df = pd.read_excel("test_automation/qna.xlsx")
#
#     questions_list = df['질의'].tolist()
#     model_answers_list = df['모범 응답'].tolist()
#
#     for question, model_answer in zip(questions_list, model_answers_list):
#         # response = self.chatbot.db_qna(llm, db, question)  # 기본 검색기
#         response = self.chatbot.db_qna_ensemble(llm, bm_db, db,
#                                                 question)  # 앙상블 검색기 (키워드 기반 문서 검색, 의미적 유사성 기반 문서 검색)
#         # response = self.chatbot.db_qna_selfQuery(llm, db, question)  # 앙상블 검색기 (셀프 쿼리 기반 문서 검색, 의미적 유사성 기반 문서 검색,)
#
#         # 코사인 유사도 확인
#         temp_model_answer = embedding_model.embed_query(model_answer)
#         temp_response = embedding_model.embed_query(response)
#         similarity = self.cosine_similarity(temp_model_answer, temp_response)
#         print(f"similarity: {similarity}")
#
#         # 파일 저장
#         # save_qna_list(question, response, model_num, similarity)
#         self.save_qna_list_v2(question, response, model_answer, model_num, similarity, embedding_model_name,
#                               chunk_size, overlap_size)


def run():
    """
    질의응답 시스템 생성 -> 환경변수 로드 -> 거대언어모델 선택 -> 검사할 파일 읽기 -> 파일 내용 읽기 -> 정답과 모델 답안 비교 -> 결과 출력
    """

    chatbot = ChatBotSystem()

    # 환경변수 로드
    chatbot.load_env()

    # 채팅에 사용할 거대언어모델(LLM) 선택
    llm, model_num = chatbot.chat_llm()

    # 파일 읽기
    file_list = chatbot.file_read()

    for file in file_list:
        print(file)
        
        # 파일 내부 내용 읽기

    correct_answer = """
    디자인조형학과의 실기시험에 대한 정보는 다음과 같습니다:

 모집단위 및 실기종목
- 모집단위: 시각디자인, 산업디자인, 공예디자인
- 실기종목: 기초디자인, 사고의 전환

종목별 배점(비율)
- 배점 : 700점 (70%)

실기종목별 준비물, 실기주제, 화지크기 및 시험시간
1. 기초디자인
- 준비물 : 볼펜, 연필, 색연필, 수채물감 등 표현에 필요한 도구
- 실기주제 : 주제A, 주제B에서 각 1개씩 선정하여 연결된 주제를 당일 추첨 (예시: 잠자리와 저울)
- 주제A: 잠자리, 무궁화, 피망, 고양이, 양파, 달팽이, 오렌지
- 주제B: 악보, 줄자, 스카치테이프, 자물쇠, 반코팅장갑, 저울, 얼음
- 조건: 제공된 사물과 이미지 이외의 형상은 표현하지 마시오. 제공된 사물은 모두 1회 이상 표현하시오. 제공된 사물의 색상은 변경하지 마시오.
- 화지크기 : 켄트지 3절
- 시험시간 : 5시간

2.사고의 전환
- 준비물: 볼펜, 연필, 색연필, 수채물감 등 표현에 필요한 도구
- 실기주제 : 주제A, 주제B에서 각 1개씩 선정하여 연결된 주제를 당일 추첨 (예시: 콜라병과 계절)
- 주제A: 콜라병, 포카리스웨트(캔), 마스크, 시계, 맥가이버 칼
- 주제B: 바이러스, 자유, 여행, 계절, 꿈
- 화지크기: 켄트지 2절
- 시험시간: 5시간

이와 같은 정보는 디자인조형학과 실기시험을 준비하는 데 중요한 요소들입니다. 추가적인 정보나 문의사항이 있다면 "https://ipsi.deu.ac.kr/main.do"를 방문하시기 바랍니다.
    """

    answer = """
    " 물론이죠, 디자인조형학과 실기시험에 대해 자세히 설명드리겠습니다!

디자인조형학과는 시각예술과 디자인을 전문으로 하는 학과로, 다양한 예술적 표현 기법을 통해 창의력과 문제 해결 능력을 개발하는 데 중점을 두고 있습니다. 이 학과의 학생들은 그래픽 디자인, 산업 디자인, 일러스트레이션 등 여러 분야에서 일할 수 있는 기술을 습득하게 됩니다.

디자인조형학과에 입학하기 위한 실기시험은 일반적으로 시각예술과 디자인 분야의 기술과 지식을 평가하는 것을 목적으로 합니다. 시험의 구체적인 내용은 대학이나 기관마다 다를 수 있지만, 대부분의 경우 다음과 같은 요소들을 평가합니다:

1. 드로잉 및 페인팅 기술: 학생들은 다양한 도구와 기법을 사용하여 정교한 드로잉과 페인팅을 제작할 능력을 보여줘야 합니다. 이는 선, 형태, 질감, 색채의 이해뿐만 아니라 관찰력, 창의성, 표현력을 평가하는 것을 포함합니다.
2. 디자인 원칙: 학생들은 시각적 요소(예: 형태, 크기, 공간, 균형)와 원근법, 비례, 대칭과 같은 원리를 사용하여 시각적으로 매력적이고 기능적인 디자인을 만드는 능력을 입증해야 합니다.
3. 문제 해결 능력: 학생들은 주어진 문제에 대한 창의적인 솔루션을 개발할 수 있는 능력을 보여주어야 하며, 이는 디자인 과정의 다양한 단계(예: 개념화, 계획 수립, 실행)를 포함합니다.
4. 미디어 이해도: 학생들은 다양한 예술적 표현 기법과 매체(예: 드로잉 도구, 페인트, 디지털 소프트웨어)에 대한 지식을 입증해야 합니다. 이는 재료의 특성, 적절한 사용 방법 및 다른 기법과의 통합을 이해하는 것을 포함합니다.
5. 창의성과 독창성: 학생들은 자신의 아이디어를 시각적으로 표현하고 디자인 솔루션을 개발할 때 혁신적인 접근 방식을 보여줄 수 있어야 합니다. 이는 다양한 스타일과 미학을 실험하는 것과 같은 능력을 평가하는 것을 포함합니다.
6. 시간 관리 및 압박감 하에서의 작업 능력: 시험은 제한된 시간 내에 작업을 완료하도록 요구하며, 학생들은 이러한 환경에서 효과적으로 작업하고 마감 기한을 지킬 수 있는 능력을 입증해야 합니다.
7. 커뮤니케이션 기술: 디자인조형학과는 시각적 언어를 통해 아이디어를 전달하는 것을 포함하므로, 학생들은 자신의 디자인을 명확하고 설득력 있게 제시할 수 있어야 합니다. 이는 포트폴리오 준비와 프레젠테이션 기술을 평가하는 것을 포함합니다.

디자인조형학과 실기시험은 일반적으로 몇 시간 동안 진행되며, 학생들이 다양한 과제를 완료하도록 요구됩니다. 시험의 구체적인 내용은 대학이나 기관에 따라 다를 수 있지만, 대부분의 경우 시각예술과 디자인 분야의 기술과 지식을 평가하기 위한 공통된 요소들을 포함하고 있습니다."

    """

    # answer = """
    #     디자인조형학과의 실기시험에 대한 정보는 다음과 같습니다:
    #
    #      모집단위 및 실기종목
    #     - 모집단위: 시각디자인, 산업디자인, 공예디자인
    #     - 실기종목: 기초디자인, 사고의 전환
    #
    #     종목별 배점(비율)
    #     - 배점 : 700점 (70%)
    #
    #     실기종목별 준비물, 실기주제, 화지크기 및 시험시간
    #     1. 기초디자인
    #     - 준비물 : 볼펜, 연필, 색연필, 수채물감 등 표현에 필요한 도구
    #     - 실기주제 : 주제A, 주제B에서 각 1개씩 선정하여 연결된 주제를 당일 추첨 (예시: 잠자리와 저울)
    #     - 주제A: 잠자리, 무궁화, 피망, 고양이, 양파, 달팽이, 오렌지
    #     - 주제B: 악보, 줄자, 스카치테이프, 자물쇠, 반코팅장갑, 저울, 얼음
    #     - 조건: 제공된 사물과 이미지 이외의 형상은 표현하지 마시오. 제공된 사물은 모두 1회 이상 표현하시오. 제공된 사물의 색상은 변경하지 마시오.
    #     - 화지크기 : 켄트지 3절
    #     - 시험시간 : 5시간
    #
    #     2.사고의 전환
    #     - 준비물: 볼펜, 연필, 색연필, 수채물감 등 표현에 필요한 도구
    #     - 실기주제 : 주제A, 주제B에서 각 1개씩 선정하여 연결된 주제를 당일 추첨 (예시: 콜라병과 계절)
    #     - 주제A: 콜라병, 포카리스웨트(캔), 마스크, 시계, 맥가이버 칼
    #     - 주제B: 바이러스, 자유, 여행, 계절, 꿈
    #     - 화지크기: 켄트지 2절
    #     - 시험시간: 5시간
    #
    #     이와 같은 정보는 디자인조형학과 실기시험을 준비하는 데 중요한 요소들입니다. 추가적인 정보나 문의사항이 있다면 "https://ipsi.deu.ac.kr/main.do"를 방문하시기 바랍니다.
    #     """

    # answer = """
    #         디자인조형학과의 실기시험은 시각디자인, 산업디자인, 공예디자인 세 가지 종목으로 구성되어 있습니다. 각 종목은 기초디자인, 사고의 전환 두 가지 주제로 이루어져 있습니다. 시험 시간은 5시간이며, 화지크기는 켄트지3절입니다. 준비물로는 볼펜, 연필, 색연필, 수채물감 등 표현에 필요한 도구가 필요합니다. 시험에서는 주제A와 주제B에서 각 1개씩 선정하여 연결된 주제를 당일 추첨하며, 주어진 사물과 이미지를 활용하여 표현해야 합니다.
    #         """

    # response = chatbot.qna(llm, correct_answer, answer)


    # print(response)


if __name__ == "__main__":
    run()
