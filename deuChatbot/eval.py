from langchain_teddynote.community.kiwi_tokenizer import KiwiTokenizer
from konlpy.tag import Okt

from rouge_score import rouge_scorer

from nltk.translate.bleu_score import sentence_bleu

import nltk
from nltk.translate import meteor_score

from sentence_transformers import SentenceTransformer, util
import torch
from torch.nn.functional import cosine_similarity

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from dotenv import load_dotenv
import warnings

import pandas as pd

import os
from datetime import datetime
from pathlib import Path

class OktTokenizer:
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text, type="list"):
        if type == "list":
            return [token for token in self.okt.morphs(text)]
        else:
            return " ".join([token for token in self.okt.morphs(text)])


def read_excel(file_path):
    """
    엑셀 파일을 읽어서 데이터프레임으로 변환하는 함수.
    질의, 응답, 모범 응답, LLM 종류를 각각 리스트로 반환.
    :param file_path: 엑셀 파일 경로
    :return: 질의 리스트, 응답 리스트, 모범 응답 리스트, LLM 종류 리스트
    """
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path)

    # 필요한 컬럼에서 데이터 추출
    llm_types = df.iloc[:, 0].tolist()  # A열
    question = df.iloc[:, 3].tolist()  # D열
    model_responses = df.iloc[:, 4].tolist()  # E열
    responses = df.iloc[:, 5].tolist()  # F열

    print(f"\nProcessing file: {file_path}")
    print("=======LLM Types=========")
    print(llm_types)
    print("=======LLM Types=========")

    print("\n=======question=========")
    print(question)
    print("=======question=========")

    print("\n========model_responses========")
    print(model_responses)
    print("========model_responses========")

    print("\n=======responses=========")
    print(responses)
    print("=======responses=========")

    print("\n")

    return question, responses, model_responses, llm_types


def rouge_score_to_excel(responses, model_responses, llm_types, output_file):
    """
    responses와 model_responses에 담긴 문자열 리스트를 순차적으로 비교하여 ROUGE 점수를 계산하고
    각 항목의 ROUGE1, ROUGE2, ROUGEL 점수와 LLM 종류별 평균을 엑셀 파일로 저장하는 함수.
    """
    scorer = rouge_scorer.RougeScorer(
        # ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=KiwiTokenizer()
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=OktTokenizer()
    )

    scores_data = []

    # 각 항목별 점수 계산
    for i, (response, model_response, llm_type) in enumerate(zip(responses, model_responses, llm_types)):
        rouge1_score = scorer.score(response, model_response)['rouge1'].fmeasure
        rouge2_score = scorer.score(response, model_response)['rouge2'].fmeasure
        rougeL_score = scorer.score(response, model_response)['rougeL'].fmeasure

        scores_data.append([llm_type, response, model_response, rouge1_score, rouge2_score, rougeL_score])

    # DataFrame 생성
    df = pd.DataFrame(scores_data, columns=['LLM Type', 'Response', 'Model Response', 'ROUGE1', 'ROUGE2', 'ROUGEL'])

    # LLM 종류별 평균 계산
    llm_averages = []
    for llm_type in df['LLM Type'].unique():
        llm_data = df[df['LLM Type'] == llm_type]
        rouge1_avg = llm_data['ROUGE1'].mean()
        rouge2_avg = llm_data['ROUGE2'].mean()
        rougeL_avg = llm_data['ROUGEL'].mean()
        llm_averages.append([f"{llm_type} Average", 'N/A', 'N/A', rouge1_avg, rouge2_avg, rougeL_avg])

    # 전체 평균 계산
    total_rouge1_avg = df['ROUGE1'].mean()
    total_rouge2_avg = df['ROUGE2'].mean()
    total_rougeL_avg = df['ROUGEL'].mean()
    llm_averages.append(['Total Average', 'N/A', 'N/A', total_rouge1_avg, total_rouge2_avg, total_rougeL_avg])

    # 평균 데이터를 DataFrame에 추가
    avg_df = pd.DataFrame(llm_averages, columns=df.columns)
    df = pd.concat([df, avg_df], ignore_index=True)

    df.to_excel(output_file, index=False)
    print(f"ROUGE scores and averages saved to {output_file}")


def bleu_score_to_excel(responses, model_responses, llm_types, output_file):
    """
    responses와 model_responses에 담긴 문자열 리스트를 순차적으로 비교하여 BLEU 점수를 계산하고
    각 항목의 BLEU 점수와 LLM 종류별 평균을 엑셀 파일로 저장하는 함수.
    """
    # kiwi_tokenizer = KiwiTokenizer()
    okt_tokenizer = OktTokenizer()
    scores_data = []

    for i, (response, model_response, llm_type) in enumerate(zip(responses, model_responses, llm_types)):
        # reference = [kiwi_tokenizer.tokenize(model_response, type="sentence")]
        # candidate = kiwi_tokenizer.tokenize(response, type="sentence")
        reference = [okt_tokenizer.tokenize(model_response, type="sentence")]
        candidate = okt_tokenizer.tokenize(response, type="sentence")
        bleu = sentence_bleu(reference, candidate)
        scores_data.append([llm_type, response, model_response, bleu])

    df = pd.DataFrame(scores_data, columns=['LLM Type', 'Response', 'Model Response', 'BLEU Score'])

    # LLM 종류별 평균 계산
    llm_averages = []
    for llm_type in df['LLM Type'].unique():
        llm_data = df[df['LLM Type'] == llm_type]
        bleu_avg = llm_data['BLEU Score'].mean()
        llm_averages.append([f"{llm_type} Average", 'N/A', 'N/A', bleu_avg])

    # 전체 평균 계산
    total_bleu_avg = df['BLEU Score'].mean()
    llm_averages.append(['Total Average', 'N/A', 'N/A', total_bleu_avg])

    avg_df = pd.DataFrame(llm_averages, columns=df.columns)
    df = pd.concat([df, avg_df], ignore_index=True)

    df.to_excel(output_file, index=False)
    print(f"BLEU scores and averages saved to {output_file}")


def meteor_score_to_excel(responses, model_responses, llm_types, output_file):
    """
    responses와 model_responses에 담긴 문자열 리스트를 순차적으로 비교하여 METEOR 점수를 계산하고
    각 항목의 METEOR 점수와 LLM 종류별 평균을 엑셀 파일로 저장하는 함수.
    """
    nltk.download('wordnet')
    # kiwi_tokenizer = KiwiTokenizer()
    okt_tokenizer = OktTokenizer()
    scores_data = []

    for i, (response, model_response, llm_type) in enumerate(zip(responses, model_responses, llm_types)):
        # reference = [kiwi_tokenizer.tokenize(model_response, type="list")]
        # candidate = kiwi_tokenizer.tokenize(response, type="list")
        reference = [okt_tokenizer.tokenize(model_response, type="list")]
        candidate = okt_tokenizer.tokenize(response, type="list")
        meteor = meteor_score.meteor_score(reference, candidate)
        scores_data.append([llm_type, response, model_response, meteor])

    df = pd.DataFrame(scores_data, columns=['LLM Type', 'Response', 'Model Response', 'METEOR Score'])

    # LLM 종류별 평균 계산
    llm_averages = []
    for llm_type in df['LLM Type'].unique():
        llm_data = df[df['LLM Type'] == llm_type]
        meteor_avg = llm_data['METEOR Score'].mean()
        llm_averages.append([f"{llm_type} Average", 'N/A', 'N/A', meteor_avg])

    # 전체 평균 계산
    total_meteor_avg = df['METEOR Score'].mean()
    llm_averages.append(['Total Average', 'N/A', 'N/A', total_meteor_avg])

    avg_df = pd.DataFrame(llm_averages, columns=df.columns)
    df = pd.concat([df, avg_df], ignore_index=True)

    df.to_excel(output_file, index=False)
    print(f"METEOR scores and averages saved to {output_file}")

def cos_similarity(a, b):
    """
    코사인 유사도를 확인하기 위한 함수
    :param a: 벡터 a
    :param b: 벡터 b
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def sem_score_to_excel(questions, responses, model_responses, llm_types, output_file):
    """
    responses와 model_responses에 담긴 문자열 리스트를 순차적으로 비교하여 코사인 유사도를 계산하고
    각 항목의 유사도 점수와 LLM 종류별 평균을 엑셀 파일로 저장하는 함수.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 임베딩 모델 사용하기
    # model = SentenceTransformer("all-mpnet-base-v2")
    # model = SentenceTransformer("nlpai-lab/KURE-v1")
    #
    # OpenAI()일 때
    # load_dotenv('.env')
    # model = OpenAIEmbeddings(
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     model='text-embedding-3-small'
    #     # model='text-embedding-3-large'
    # )

    # HuggingFaceEmbeddings 일때
    # model_name = "jhgan/ko-sbert-nli"
    # model_name = "jhgan/ko-sroberta-multitask"
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = "intfloat/multilingual-e5-large"
    # model_name = "nlpai-lab/KURE-v1"
    # model_name = "nlpai-lab/KoE5"
    model_name = "BAAI/bge-m3"

    # model_kwargs = {'device': 'cpu'}  # cpu를 사용하기 위해 설정
    model_kwargs = {'device': 'cuda'}  # gpu를 사용하기 위해 설정
    encode_kwargs = {'normalize_embeddings': True}

    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("여기까지 옴")

    scores_data = []

    for i, (question, response, model_response, llm_type) in enumerate(zip(questions, responses, model_responses, llm_types)):
        q = str(question)
        response = str(response)
        model_response = str(model_response)

        # 이 부분에 임베딩 모델 바꿔서 넣기
        # SentenceTransformer("all-mpnet-base-v2")
        # response_encoded = model.encode(response, convert_to_tensor=True)
        # model_response_encoded = model.encode(model_response, convert_to_tensor=True)
        # response_encoded = model.encode(response)
        # model_response_encoded = model.encode(model_response)

        print(q)
        print(response)
        print(model_response)

        # OpenAI(), HuggingFace 일 때
        question_encoded = model.embed_query(q)
        response_encoded = model.embed_query(response)
        model_response_encoded = model.embed_query(model_response)

        question_encoded = torch.tensor(question_encoded)
        response_encoded = torch.tensor(response_encoded)
        model_response_encoded = torch.tensor(model_response_encoded)

        question_encoded = question_encoded.unsqueeze(0)
        response_encoded = response_encoded.unsqueeze(0)
        model_response_encoded = model_response_encoded.unsqueeze(0)

        print(question_encoded)
        print(response_encoded)
        print(model_response_encoded)

        # cosine_similarity = cos_similarity(response_encoded, model_response_encoded)
        # cosine_sim = cosine_similarity(response_encoded, model_response_encoded).item()

        # 질문 - 응답 유사도 비교
        cosine_sim = cosine_similarity(response_encoded, model_response_encoded).item()
        # cosine_sim = cosine_similarity(question_encoded, model_response_encoded).item()
        print(cosine_sim)

        # cosine_sim = util.pytorch_cos_sim(response_encoded, model_response_encoded).item()
        # print(cosine_sim)
        scores_data.append([llm_type, response, model_response, cosine_sim])

    df = pd.DataFrame(scores_data, columns=['LLM Type', 'Response', 'Model Response', 'Cosine Similarity'])
    # df = pd.DataFrame(scores_data, columns=['LLM Type', 'Question', 'Model Response', 'Cosine Similarity'])

    # LLM 종류별 평균 계산
    llm_averages = []
    for llm_type in df['LLM Type'].unique():
        llm_data = df[df['LLM Type'] == llm_type]
        similarity_avg = llm_data['Cosine Similarity'].mean()
        llm_averages.append([f"{llm_type} Average", 'N/A', 'N/A', similarity_avg])

    # 전체 평균 계산
    total_similarity_avg = df['Cosine Similarity'].mean()
    llm_averages.append(['Total Average', 'N/A', 'N/A', total_similarity_avg])

    avg_df = pd.DataFrame(llm_averages, columns=df.columns)
    df = pd.concat([df, avg_df], ignore_index=True)

    df.to_excel(output_file, index=False)
    print(f"Cosine similarity scores and averages saved to {output_file}")


def process_folder(input_folder):
    """
    지정된 폴더 내의 모든 엑셀 파일을 처리하는 함수
    :param input_folder: 입력 엑셀 파일이 있는 폴더 경로
    """

    # 각 메트릭별 결과 폴더 생성
    base_output_folder = f"{input_folder}/results"

    output_folders = {
        'sem': f"{base_output_folder}/sem_score",
        'meteor': f"{base_output_folder}/meteor_score",
        'bleu': f"{base_output_folder}/bleu_score",
        'rouge': f"{base_output_folder}/rouge_score"
    }

    # 모든 결과 폴더 생성
    for folder in output_folders.values():
        Path(folder).mkdir(parents=True, exist_ok=True)

    # 입력 폴더의 모든 엑셀 파일 처리
    for filename in os.listdir(input_folder):
        print(filename)
        if filename.endswith('.xlsx'):
            input_path = os.path.join(input_folder, filename)

            # 파일 처리
            question, responses, model_responses, llm_types = read_excel(input_path)

            # 문자열로 변환
            question = [str(q) for q in question]
            responses = [str(response) for response in responses]
            model_responses = [str(model_response) for model_response in model_responses]
            llm_types = [str(llm_type) for llm_type in llm_types]


            # 각 메트릭별 결과 파일 경로 설정
            sem_output = os.path.join(output_folders['sem'], f"sem_{filename}")
            meteor_output = os.path.join(output_folders['meteor'], f"meteor_{filename}")
            bleu_output = os.path.join(output_folders['bleu'], f"bleu_{filename}")
            rouge_output = os.path.join(output_folders['rouge'], f"rouge_{filename}")

            # 각 메트릭 계산 및 저장
            sem_score_to_excel(question, responses, model_responses, llm_types, sem_output)
            meteor_score_to_excel(responses, model_responses, llm_types, meteor_output)
            bleu_score_to_excel(responses, model_responses, llm_types, bleu_output)
            rouge_score_to_excel(responses, model_responses, llm_types, rouge_output)


if __name__ == "__main__":
    # input_folder = "research_result"  # 입력 폴더 경로
    # input_folder = "250109/kosbert_maxtokens(100)"  # 입력 폴더 경로
    # input_folder = "250109/kosbert_maxtokens(200)"  # 입력 폴더 경로
    # input_folder = "250109/kosbert_maxtokens(300)"  # 입력 폴더 경로
    # input_folder = "250109/kosbert_maxtokens(400)"  # 입력 폴더 경로

    # date = datetime.today().strftime('%y%m%d')
    # input_folder = f"research_result/{date}"  # 입력 폴더 경로

    # input_folder = f"research_result/gpt_4o_mini/version_2"  # 입력 폴더 경로
    input_folder = f"research_result/gpt_4o_mini/table_json"  # 입력 폴더 경로

    process_folder(input_folder)
