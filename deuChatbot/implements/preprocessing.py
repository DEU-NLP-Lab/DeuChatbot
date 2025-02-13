import pandas as pd
import json
import re


class GPTScorePreprocessing:
    def __init__(self, file_path: str, sheet_name: str, col_model_name: int, col_gpt_score: int,
                 save_path: str, save_name: str) -> None:
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.col_model_name = col_model_name
        self.col_gpt_score = col_gpt_score
        self.save_path = save_path
        self.save_name = save_name
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

                try:
                    # JSON 파싱
                    parsed_json = json.loads(text_search_json)

                    # 모든 값을 정수로 변환
                    converted_json = {k: int(v) for k, v in parsed_json.items()}

                    # 모델별로 중첩된 딕셔너리 생성
                    if current_model not in self.preprocessed_json_dict:
                        self.preprocessed_json_dict[current_model] = {}

                    # 변환된 JSON 데이터를 저장
                    self.preprocessed_json_dict[current_model][counter] = converted_json
                    counter += 1
                except (ValueError, TypeError) as e:
                    print(f"[{row_num}] 행의 JSON 값 변환 중 오류 발생: {e}")

                # # 모델별로 중첩된 딕셔너리 생성
                # if current_model not in self.preprocessed_json_dict:
                #     self.preprocessed_json_dict[current_model] = {}
                #
                # # JSON 데이터를 파싱하여 저장
                # self.preprocessed_json_dict[current_model][counter] = json.loads(text_search_json)
                # counter += 1
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

        # 일반 점수 결과 저장
        self.save_json(self.save_path, self.save_name)

    def aspect_normalize_json(self) -> None:
        """모델별 점수 평균 계산 및 정규화"""
        result = {}
        metrics_averages = {}

        # 각 모델의 데이터 처리
        for model_name, model_data in self.preprocessed_json_dict.items():
            normalized_scores = {}
            metrics_sums = {}  # 각 지표별 합계
            metrics_counts = {}  # 각 지표별 카운트

            # 각 응답에서 지표별로 점수 합산
            for idx, score_dict in model_data.items():
                for metric, score in score_dict.items():
                    if metric not in metrics_sums:
                        metrics_sums[metric] = 0
                        metrics_counts[metric] = 0
                    metrics_sums[metric] += score
                    metrics_counts[metric] += 1

            # 각 지표별 평균 계산
            for metric in metrics_sums.keys():
                avg_score = round((metrics_sums[metric] / metrics_counts[metric]) * 20, 2)
                normalized_scores[f"{metric}_average"] = avg_score

                # 지표별 전체 평균 계산을 위해 저장
                if metric not in metrics_averages:
                    metrics_averages[metric] = {}
                metrics_averages[metric][model_name] = avg_score

            result[model_name] = normalized_scores

        # 각 지표별 최고 점수 모델 찾기
        for metric in metrics_averages.keys():
            best_model, best_score = self.find_best_model(metrics_averages[metric])
            result[f"{metric}_best_model"] = f"{best_model} : {best_score}"

        self.output_json = result

        # 지표별 점수 결과 저장
        self.save_json(self.save_path, f"aspect_{self.save_name}")

    def find_best_model(self, score_dict: dict) -> tuple:
        """최고 평균 점수를 가진 모델을 반환"""
        best_model = max(score_dict, key=score_dict.get)
        return best_model, score_dict[best_model]

    def save_json(self, save_path: str, save_name: str) -> None:
        """결과를 JSON 파일로 저장"""
        with open(f"{save_path}/{save_name}.json", "w", encoding="UTF-8") as f:
            json.dump(self.output_json, f, indent=4)

    def run(self) -> None:
        """실행 메서드"""
        self.load_excel()
        self.extract_json()
        self.normalize_json()
        # self.aspect_normalize_json()

