# utils.py 일부

import json, ast

# === 채점용 프롬프트 템플릿 ===
PROMPT_TEMPLATE = (
    "너는 국어 교과 평가를 담당하는 채점자이다.\n"
    "아래 제시문을 읽고 학생의 답변이 (가)와 (나)의 목적을 얼마나 정확히 파악했는지 0~2점으로 채점하라.\n"
    "출력은 오직 \"0점\", \"1점\", \"2점\" 중 하나만 써라. 다른 설명이나 이유는 절대 쓰지 마라.\n\n"
    "[제시문 요약]\n"
    "(가): 과장·허위 광고의 문제점을 알려 주고 주의시키는 내용\n"
    "(나): '쑥쑥 주스'를 홍보하고 판매하려는 광고\n\n"
    "[채점 기준]\n"
    "2점: (가)와 (나)의 목적을 모두 정확히 파악함\n"
    "1점: 둘 중 하나만 정확히 파악함\n"
    "0점: 둘 다 틀리거나 목적을 제대로 파악하지 못함\n"
    "의미가 같으면 표현이 달라도 정답으로 인정하라.\n\n"
    "[채점결과 예시]\n"
    "0점\n\n"
    "[학생 답변]\n"
    "{response}\n\n"
    "[채점 결과]\n"
)

def _build_scoring_prompt(student_response: str) -> str:
    return PROMPT_TEMPLATE.format(response=student_response if student_response is not None else "")

def get_datasetM(file_name, args):
    """
    반환:
      texts  : 학습 입력 프롬프트 리스트 (채점 템플릿 적용)
      labels : 정답 레이블 리스트 ("0점" | "1점" | "2점")
    지원 입력 형식:
      1) [{"response": str, "point": int}, ...]  # 새 채점 데이터
      2) [{... metaData/metaIntent/dialog ...}]  # 기존 메타 구조
    """
    texts, labels = [], []

    with open(file_name, "r", encoding="utf-8") as fn:
        data = json.load(fn)

    # --- 새 채점 데이터 형태인지 판단: 'response' 키 존재 ---
    if isinstance(data, list) and data and isinstance(data[0], dict) and "response" in data[0]:
        for ex in data:
            resp = ex.get("response", "")
            point = ex.get("point", None)
            if point not in (0, 1, 2):
                # 잘못된 라벨은 스킵
                continue
            texts.append(_build_scoring_prompt(resp))
            labels.append(f"{point}점")
        return texts, labels

    # --- 기존 메타 구조 (fallback) ---
    for line in data:
        # 문자열로 들어온 경우 방어적 파싱
        if isinstance(line, str):
            try:
                line = ast.literal_eval(line)
            except Exception:
                continue

        # 학습 데이터만 사용
        try:
            if line["metaData"][0]["dataType"] != "Train":
                continue
        except Exception:
            continue

        # 메타 인텐트 괄호 포함 샘플 스킵(기존 규칙 유지)
        meta_intent = line.get("metaIntent", "")
        if "(" in meta_intent:
            continue

        # 프롬프트는 채점 템플릿이 아니라 기존 대화 템플릿을 원하면 아래를 바꿔도 됨.
        # 여기서는 새 채점 템플릿으로 동일 적용: 학생 답변 = 첫 발화 utterance 로 가정
        try:
            student_resp = line["dialog"][0]["utterance"]
        except Exception:
            student_resp = ""

        texts.append(_build_scoring_prompt(student_resp))

        # 기존 메타 라벨을 0/1/2로 매핑할 규칙이 없다면, 일단 그대로 사용하지 않고 스킵하거나
        # 필요 시 규칙을 작성해야 함. 여기선 그대로 문자열을 "라벨"로 넣지 않고
        # 안전하게 무시. (원한다면 매핑 규칙을 제공해줘!)
        # labels.append(meta_intent)

    return texts, labels
