import pandas as pd
import json
import re


class GPTScorePreprocessing:
    def __init__(self, file_path: str, sheet_name: str, col_model_name: int, col_gpt_score: int) -> None:
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.col_model_name = col_model_name
        self.col_gpt_score = col_gpt_score
        self.df: pd.DataFrame = None
        self.len_row: int = None
        self.json_pattern = r'\{.*?\}'
        self.preprocessed_json_dict = {}
        self.reset_data()
        

    def reset_data(self) -> None:
        """데이터 초기화 메서드"""
        self.preprocessed_json_dict = {}
        self.output_json = {}

    def load_excel(self) -> None:
        """엑셀 파일 로드 및 초기화"""
        self.reset_data()
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        self.len_row = self.df.shape[0]

    def extract_json(self) -> None:
        """각 행에서 JSON 데이터 추출 및 전처리"""
        counter = 1
        
        for row_num in range(self.len_row):
            current_model = self.df.iloc[row_num, self.col_model_name]

            # 이전 모델과 비교하여 다르면 카운터 초기화
            if row_num > 0 and current_model != self.df.iloc[row_num - 1, self.col_model_name]:
                counter = 1

            text_gpt_score = self.df.iloc[row_num, self.col_gpt_score]
            search_json = re.search(self.json_pattern, text_gpt_score, re.DOTALL)

            if search_json:
                text_search_json = search_json.group()

                # 모델별로 중첩된 딕셔너리 생성
                if current_model not in self.preprocessed_json_dict:
                    self.preprocessed_json_dict[current_model] = {}

                # JSON 데이터를 파싱하여 저장
                self.preprocessed_json_dict[current_model][counter] = json.loads(text_search_json)
                counter += 1
            else:
                print(f"[{row_num}] 행에서 JSON 형식이 발견되지 않았습니다.")

    def normalize_json(self) -> None:
        """모델별 점수 평균 계산 및 정규화"""
        result = {}
        aver_scores = {}

        for model_name, model_data in self.preprocessed_json_dict.items():
            total_score = 0
            normalized_scores = {}

            for idx, score_dict in model_data.items():
                score_sum = sum(score_dict.values())
                score_count = len(score_dict)
                normalized_score = round((score_sum / score_count) * 20)
                normalized_scores[idx] = normalized_score
                total_score += normalized_score

            avg_score = round(total_score / len(model_data), 2)
            normalized_scores["average"] = avg_score
            result[model_name] = normalized_scores
            aver_scores[model_name] = avg_score

        # 최고 평균 점수를 가진 모델 찾기
        best_model, best_score = self.find_best_model(aver_scores)
        result["high score model"] = f"{best_model} : {best_score}"
        self.output_json = result

    def find_best_model(self, score_dict: dict) -> tuple:
        """최고 평균 점수를 가진 모델을 반환"""
        best_model = max(score_dict, key=score_dict.get)
        return best_model, score_dict[best_model]

    def save_json(self, save_path: str, save_name: str) -> None:
        """결과를 JSON 파일로 저장"""
        with open(f"{save_path}/{save_name}.json", "w", encoding="UTF-8") as f:
            json.dump(self.output_json, f, indent=4)
