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
1. 문서 임베딩에 사용할 모델을 선택한다.
    * OpenAIEmbedding 경우 (유료)
        ```python
         os.environ['OPENAI_API_KEY'] = input('발급 받은 OpenAI API Key를 입력해주세요: ')
        ```
        OpenAI API Key를 입력하면 정상 동작한다.   
        단, 잔여 크레딧이 있을 경우에만 실행 가능하다.
    * HuggingFaceEmbedding 경우 (무료)
        API Key 상관없이 무료로 바로 사용 가능하다.
2. 채팅에 사용할 모델을 선택한다.
   * ChatOpenAI 경우 (유료)
   * Google Gemini Pro 경우 (무료)
     1. https://aistudio.google.com 접속하여 가입한다.
     2. 첨부된 사진과 같이 좌측에 있는 Get API Key를 선택하고 Create API key를 누르면 발급된다.   
      ![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/644bd75b-9719-442b-928f-9faadd6b0642)
     3. 발급받은 API Key를 입력한다.
   * Ollama 경우 (무료)   
     1. 선택 전 https://ollama.com/ 사이트에 접속하여 다운로드한다.   
     2. GGUF: https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF   
        사이트에 접속하여 file을 선택하고 ggml-model-Q4_K_M.gguf 파일을 챗봇 폴더에 다운로드 한다.   
     3. cmd 창을 실행한다.   
        ```ollama create EEVE-Korean-10.8B -f EEVE-Korean-Instruct-10.8B-v1.0-GGUF/Modelfile```   
     4. Ollama 모델 목록을 확인한다.   
        ```ollama list```
     5. Ollama 모델을 실행한다.   
        ```ollama run EEVE-Korean-10.8B:latest```
        
3. 알고싶은 정보를 질문한다.
4. 프로그램을 계속 사용할지 프로그램을 종료할지 선택한다.

- - -
결과
----------
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/bbcf17be-4a88-4e53-b718-e3c52debd13e)   
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/85366d26-f41c-48f6-a5f6-2d4b480ce9cb)   
![image](https://github.com/915-Lab/DeuChatbot/assets/93813747/f4b36226-8464-47b8-9f1d-0a905ff38b3b)   
![image](https://github.com/915-Lab/DBToJsonProject/assets/138217806/c85b0ab3-fa8f-4ada-96ae-2326552a88e6)
- - -
추가 예정 기능
----------
* 거대 언어 모델 오픈 소스 기반으로 변경 예정