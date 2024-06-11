# LangChain을 이용한 거대언어모델(LLM) 기반의 질의응답 시스템
## OO대학교 정시 입시 데이터를 기반으로 구축된 RAG 챗봇
- - -
## 목차
[1) 개발 목적](#1-개발-목적)   
[2) 시스템 설계](#2-시스템-설계)   
[3) 말뭉치 구축](#3-말뭉치-구축)  
[4) 시스템 구축](#4-시스템-구축)  
[5) 실행 방법](#5-실행-방법)  
[6) 실행 화면](#6-실행-화면)  
[7) 실험 전략](#7-실험-전략)  
[8) 실험 결과](#8-실험-결과)  
[9) 결론](#9-결론)
- - -

## 1) 개발 목적
1. LangChain Framework를 이용하여 거대언어모델의 세 가지 한계점을 보완하고자 한다.
2. 미세조정(Fine-tuning) 기법과 In-context Learning 기법을 비교하고자 한다.
3. 미세조정된 기존 질의응답시스템과 LangChain을 이용하여 구축한 질의응답시스템을 비교하고자 한다.

### 1.1) 거대언어모델 한계점
- 정보 접근 제한
- 토큰 제한
- 작화 현상

### 1.2) 한계점 보완을 위한 기법
- 미세조정(Fine-tuning)
- N-shot Learning
- In-context Learning

### 1.3) 비교군 스펙
- 미세조정(Fine-tuning) 기법 사용
- 거대언어모델 (GPT-3.5-turbo)
- 데이터셋 - OO대학교 정시 입시 모집 요강(pdf or HTML), 입시 결과(pdf or HTML)

- - -
## 2) 시스템 설계
### 2.1) 시스템 구조도 
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/2ad806ef-fdda-418d-81fd-defc6d233b89)
- RAG
  1. Document: 데이터셋을 읽어들인다.
  2. TextSplitter: 문서를 설정한 청크 크기로 분할한다.
  3. Embedding: 분할된 문서를 임베딩한다.
  4. VectorStore: 임베딩된 문서를 저장한다.
  5. Question: 사용자의 질문을 입력받는다.
  6. Embedding: 사용자의 질문을 임베딩한다.
  7. Retriever: 사용자의 질문과 가장 유사한 문서를 찾는다.
  8. Answer: 찾은 문서를 기반으로 사용자의 질문에 대답한다.

### 2.2) 한계점 보완 방법 설계
공통: LangChain Framework를 이용한 RAG 시스템을 구축하여 세 가지 한계점 보완
- 정보 접근 제한
    - 데이터셋(OO대학교 정시 입시 데이터 및 입시 결과)을 기반으로 질문에 대답하도록 구현
- 토큰 제한
    - 문서 분할기(TextSplitter)를 통한 토큰 제한 한계점 보완
- 작화 현상
    - RAG 구조를 이용하여 데이터셋을 기반으로 대답하도록 함으로써 작화 현상 보완

### 2.3) 데이터셋
- OO대학교 정시 입시 데이터
    - 정시 입시 데이터: OO대학교 정시 입시 모집 요강
    - 정시 입시 결과 데이터: OO대학교 정시 입시 결과
- - -
## 3) 말뭉치 구축
거대언어모델은 자연어 처리에서는 성능이 우수하지만, 정형데이터에 대한 성능은 떨어진다. 
따라서, 정형데이터를 잘 처리하기 위한 전처리 기법을 적용하여 말뭉치를 구축한다.
비교군(듀듀 챗봇) 질의응답 시스템은 전처리 기법을 적용하지 않았다.
정형데이터 인식률을 높이는 전처리 기법은 추후 공개 예정이다.

- 말뭉치
  - 표 75개
  - 평균 속성 3.84개
  - 평균 행 6.76개

### 3.1) 전처리 기법 적용 전 형태
![image](https://private-user-images.githubusercontent.com/93813747/316488325-4731972d-16fb-4434-8533-6655ae89d7ab.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTgwODYzMzgsIm5iZiI6MTcxODA4NjAzOCwicGF0aCI6Ii85MzgxMzc0Ny8zMTY0ODgzMjUtNDczMTk3MmQtMTZmYi00NDM0LTg1MzMtNjY1NWFlODlkN2FiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjExVDA2MDcxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJmM2QwZTllODNkOGU4YzAwY2ZhMDc0MDk3OTkwMmQ0ODIxMjdlMjA4Mjg5NDUyOWVlM2QxM2ZkNjY2ODc0NjYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.LoB-da4rsYmrygtpVMyR82Qv5XpVE9Zy65InOBrX11k)

### 3.2) 전처리 기법 적용 후 형태
![image](https://private-user-images.githubusercontent.com/93813747/316865221-e1dc15b8-c69e-4ff5-afe1-b079853f834a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTgwODYzMzgsIm5iZiI6MTcxODA4NjAzOCwicGF0aCI6Ii85MzgxMzc0Ny8zMTY4NjUyMjEtZTFkYzE1YjgtYzY5ZS00ZmY1LWFmZTEtYjA3OTg1M2Y4MzRhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjExVDA2MDcxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVlNGFlOGRjZjlhZjNmNDcyMTEwZmZkZGM4M2Y3MGZmYzNkMDM4YjE3ZGNhNjY5ZmRmYmQ0MDc0ZGRjYTQ3ZmEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.zhsTHcAj2P-9Km7YOxsDiBrTaC8dsoXQpaJAQ0jnAfg)

- - -

## 4) 시스템 구축
### 4.1) 거대언어모델 종류

- Closed Source
  - OpenAI (유료)
    - GPT-3.5-turbo
    - GPT-4-turbo
    - GPT-4o
  - Anthropic (유료)
    - Claude-3-sonnet
    - Claude-3-opus
  - Google (무료)
    - Gemini 1.5 Pro
- Open Source (HuggingFace)  
  - Yanolja
    - EEVE-Korean-Instruct-10.8B-v1.0
  - Maum-AI
    - Llama-3-MAAL-8B-Instruct-v0.1
  - Qwen
    - Qwen1.5-14B-Chat

### 4.2) 문서 임베딩모델 종류
- OpenAIEmbedding
  - text-embedding-3-small
  - text-embedding-3-large
  - text-embedding-ada-002
- HuggingFaceEmbedding
  - beomi
    - KcELECTRA-base
    - KcELECTRA-small
    - kcbert-base
  - jhgan
    - ko-sroberta-multitask
    - ko-sbert-multitask
    - ko-sroberta-nli
    - ko-sbert-nli
    - ko-sroberta-sts
    - ko-sbert-sts
  - Dongjin-kr
    - ko-reranker
  - BM-K
    - KoSimCSE-roberta-multitask 
  - sentence-transformers
    - paraphrase-multilingual-MiniLM-L12-v2
    - paraphrase-multilingual-mpnet-base-v2
- - -

## 5) 실행 방법
### 5.1) 환경 설정
#### 5.1.1) 설치 프로그램
1. Python 3.10.x
2. PyCharm
3. Git
4. LM Studio
#### 5.1.2) 패키지 설치
```
pip install -r requirements.txt
```
#### 5.1.3) deuChatbot > .env 파일 생성
```
OPENAI_API_KEY="your OpenAI API Key"
ANTHROPIC_API_KEY="your Anthropic API Key"
GOOGLE_API_KEY="your Google API Key"
UPSTAGE_API_KEY="your Upstage API Key"

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your LangChain API Key"

LM_URL="your LM Studio Server URL"
LM_LOCAL_URL="http://localhost:1234/v1"
```
### 5.2) 임베딩모델 선택
* OpenAIEmbedding 
  * OpenAI API 호출에 필요한 양의 크레딧이 필요하다.
* HuggingFaceEmbedding
  * 무료이기 때문에 바로 사용 가능하다.
### 5.3) 거대언어모델 선택
* Closed Source
  * 잔여 크레딧이 있을 경우 API 요청을 통해 사용 가능하다.
* Open Source (HuggingFace)
  * LM Studio에 아래 거대언어모델 목록의 GGUF 파일 다운로드 하여 실행하기 
    * Yanolja
      * EEVE-Korean-Instruct-10.8B-v1.0
    * Maum-AI
      * Llama-3-MAAL-8B-Instruct-v0.1
    * Qwen
      * Qwen1.5-14B-Chat
### 5.4) 입시 관련 질문 (수동 or 자동) - 자동의 경우 응답 결과가 엑셀로 저장됨
### 5.5) 프로그램 종료 후 결과 확인

- - -
## 6) 실행 화면
### 6.1) 임베딩모델 선택
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/260fc1e2-c47b-4570-8987-3e4f747f2723)

### 6.2) 거대언어모델 선택
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/173b1c99-89e0-4d21-ba9b-1b2123403a71)

### 6.3) 질문 입력 방식 선택 (수동 or 자동)
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/e5e27564-db12-419b-8a46-88043039cd0e)
#### 6.3.1) 수동 입력
질문 내용을 입력하면 거대언어모델을 통해 응답 받을 수 있다.
#### 6.3.2) 자동 입력
1. 입시 관련 질문, 모범 응답을 qna.xlsx 파일로 저장 후 프로그램을 실행한다.
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/5bf7f791-3858-432c-96fc-0b263d10542e)
2. 결과 파일 엑셀 저장(거대언어모델을 통한 응답과 모범 응답의 유사도, 질의, 응답, 모범 응답)
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/d849b1a1-8165-4ab6-9dac-7434262f4dfa)

- - -
## 7) 실험 전략
### 7.1) Corpus
- 사람이 직접 작성한 데이터셋을 사용하여 성능 비교
  - OO대학교 정시 입시 문서를 기반 질문 100개, 모범 응답 100개
    - 대학 입시 관련 커뮤니티 질문 + 모범 응답
    - 재학생 질문 + 모범 응답
    - OO대학교 입학처 직원 질문 + 모범 응답
- AI가 OO대학교 정시 입시 문서를 기반으로 생성한 데이터셋을 사용하여 성능 비교
  - AI가 생성한 질문 100개, 모범 응답 100개
    - 질문 생성에 사용될 거대언어모델: GPT-4o (2024-06-11 기준 가장 좋은 성능인 모델)
### 7.2) TextSplitter
- 청크 사이즈 조절 실험 (최적의 청크 사이즈 찾기)
- 오버랩 조절 실험

### 7.3) Embedding
- 임베딩 모델 종류 늘이기
- 임베딩 모델별 성능 비교 (한국어, 정형데이터에 최적인 임베딩 모델 찾기)

### 7.4) Retriever
기본 검색기부터 다양한 검색기를 추가했을 때 성능 변화를 확인하기 위한 실험

- Vector-store-backed retriever (기본)
- Ensemble Retriever
- Long-Context Reorder
- Self-querying
- MultiQueryRetriever
- Contextual compression
- Custom Retriever
- MultiVector Retriever
- Parent Document Retriever
- Time-weighted vector store retriever

- - -

## 8) 실험 결과


- - -

## 9) 결론
