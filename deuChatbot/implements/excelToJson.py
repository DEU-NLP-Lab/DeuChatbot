import pandas as pd
import json
import re


class ExcelToJson:
    # 생성자 및 초기화
    def __init__(self, file_path: str, sheet_name: str, col_model_name: int, col_gpt_score: int) -> None:
        self.save_name = None
        self.save_path = None
        self.df = None
        self.json_pattern = None
        self.model_name = None
        self.len_row = None
        self.json_output = None
        self.json_added_key = None
        self.result_json_search = None
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.col_model_name = col_model_name
        self.col_gpt_score = col_gpt_score
        self.reset_data()  # 초기화 메서드 호출

    def reset_data(self):
        # 데이터 초기화 메서드: 데이터를 저장하는 변수들을 초기화
        self.result_json_search = []
        self.json_added_key = {}
        self.json_output = ""

    def load_excel(self) -> None:
        self.reset_data()  # 각 파일 처리 전 데이터 초기화
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        self.len_row = self.df.shape[0]  # 최대 행 개수
        self.model_name = self.df.iloc[0, self.col_model_name]  # 모델 이름 열 데이터
        self.json_pattern = r'\{.*?\}'  # json format 추출 정규식

    # DataFrame의 최대 행 갯수까지 반복하며 Json 추출
    def extract_json(self) -> None:
        for row_num in range(self.len_row):
            try:
                text_gpt_score: str = self.df.iloc[row_num, self.col_gpt_score]
                search_json = re.search(self.json_pattern, text_gpt_score, re.DOTALL)  # 정규식 탐색
                if search_json:  # 정규식 결과가 있는지 확인
                    text_search_json: str = search_json.group()
                    self.result_json_search.append(text_search_json)
                else:
                    print(f"[ {row_num} ] 행에서 JSON 형식이 발견되지 않았습니다.")

            except Exception as e:
                print(f"[ {row_num} ] 번 행에서 오류 발생...{e}")

    def append_num_key(self) -> None:
        for i in range(self.len_row):
            try:
                self.json_added_key[i + 1] = json.loads(self.result_json_search[i])
            except IndexError:
                print(f"{i + 1} 번째 데이터가 누락되었습니다.")
            except json.JSONDecodeError as e:
                print(f"{i + 1} 번째 데이터에서 JSON 디코딩 오류 발생...{e}")

        self.json_output = json.dumps(self.json_added_key, indent=4)

    def save_json(self, save_path: str, save_name: str) -> None:
        self.save_path = save_path
        self.save_name = save_name

        with open(f"{save_path}/{save_name}.json", 'w', encoding="UTF-8") as f:
            f.write(self.json_output)
