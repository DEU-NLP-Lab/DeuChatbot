import json
import os


class JsonNormalizer:
    file_path: str
    save_path: str

    input_json: dict
    output_json: dict

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_json(self) -> None:
        with open(self.file_path, "r") as f:
            self.input_json = json.load(f)

    def normalize_json(self) -> None:
        """
        data에 있는 데이터 중 점수 딕셔너리를 합하여 새로운 딕셔너리 생성 함수
        """
        result = {}
        total_score = 0
        for key, value in self.input_json.items():  # key -> "1", "2" / value -> 점수 딕셔너리
            score_sum = sum(value.values())  # value 딕셔너리의 점수 합
            score_count = len(value)  # 점수 개수
            normalized_score = int(round((score_sum / score_count) * 20))  # 평균 계산 후 20 곱하기
            result[key] = normalized_score
            total_score += normalized_score

        # 평균값 계산
        if len(result) > 0:
            result["average"] = round((total_score / len(result)), 2)  # 평균값 계산
        self.output_json = result

    def save_json(self, save_path: str, save_name: str) -> None:
        with open(f"{save_path}/{save_name}.json", "w", encoding="UTF-8") as f:
            json.dump(self.output_json, f, indent=4)
