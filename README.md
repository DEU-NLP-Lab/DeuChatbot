# DeuChatbot - 동의대학교 입시 챗봇

개발 목적
----------
거대 언어 모델의 한계점으로는 정보 접근 제한, 토큰 제한, 환각현상이 있다.

이를 보완하기 위한 방법으로 Fine-tuning, N-shot Learning, In-context Learning 세 가지 기법이 있다.

Fine-tuning 기법과 In-context Learning 기법을 비교하기 위하여 LangChain을 이용하여 In-context
Learning 기법을 적용해 DeuChatbot을 개발하였다.

Fine-tuning 기법을 이용한 듀듀 챗봇을 비교대상으로 선정하였다.
- - -
설명
----------
본 프로그램은 전처리한 동의대 입시 데이터를 기반으로 질문에 대답해주는 챗봇 프로그램이다.  
LangChain을 이용하여 아래의 RAG 구조에 맞게 개발하였다.  

![image](https://github.com/915-Lab/DBToJsonProject/assets/138217806/32336129-ed9b-4a62-a213-9a7ed43b436d)  
- - -
패키지 버전
-----------
requirements.txt
```
pip install -r requirements.txt
```
- - -
사용방법
-----------
1. 문서 임베딩에 사용할 모델을 고른다.
2. 알고싶은 정보를 질문한다.
3. 프로그램을 계속 사용할지 프로그램을 종료할지 선택한다.

- - -
결과
----------
![image](https://github.com/915-Lab/DBToJsonProject/assets/138217806/7562046c-cad1-44c3-9c14-f4cbcbcc58bb)  
![image](https://github.com/915-Lab/DBToJsonProject/assets/138217806/e80fa0a0-ee1a-4b66-9d5b-b86a9a69e0bc)  
![image](https://github.com/915-Lab/DBToJsonProject/assets/138217806/c85b0ab3-fa8f-4ada-96ae-2326552a88e6)
- - -
추가 예정 기능
----------
* 거대 언어 모델 오픈 소스 기반으로 변경 예정