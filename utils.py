# utils.py
import os
import json
from typing import Dict, List, Tuple, Any

# =========================
#  PROMPT 템플릿
# =========================
PROMPT = (
    "너는 {subjects} 교과 평가를 담당하는 채점자이다.\n"
    "학생의 답변을 {allowed_str} 중 하나로만 채점하여 출력해라. 다른 설명이나 이유는 절대 쓰지 마라.\n"
    "[채점 기준]\n"
    "{rubric}\n"
    "의미가 같으면 표현이 달라도 정답으로 인정하라.\n\n"
    "[채점결과 예시]\n"
    "0점\n\n"
    "[학생 답변]\n"
    "{response}\n\n"
    "[채점 결과]\n"
)

# =========================
#  유틸 함수들
# =========================

def infer_subject_from_meta(raw: dict, fallback_path: str = "") -> str:
    """
    meta.source_meta.data_path 문자열에서 과목(국어/사회)을 추론.
    없으면 fallback_path로 추론, 그래도 없으면 '국어'.
    """
    dp = raw.get("meta", {}).get("source_meta", {}).get("data_path", "") or fallback_path
    if "사회" in dp:
        return "사회"
    if "국어" in dp:
        return "국어"
    return "국어"


def scan_allowed_points_by_detail(raw: dict) -> Dict[str, List[int]]:
    """
    detail_factor_id -> 허용 점수 목록(오름차순).
    같은 detail_factor_id에 대해 등장하는 point들을 모아서 허용 점수로 사용.
    (요청: allowed_str는 해당 detail_factor_id에서 나오는 점수만 사용)
    """
    mp: Dict[str, set] = {}
    for it in raw.get("data", []):
        did = it.get("detail_factor_id", "GLOBAL")
        try:
            p = int(it.get("point", 0))
        except Exception:
            p = 0
        mp.setdefault(did, set()).add(p)
    return {k: sorted(list(v)) for k, v in mp.items()}


# utils.py 내부: 요약 포맷터 교체
def _format_summaries(summaries):
    """
    summaries가 dict면:
      {"stem": "...", "constraints": ["...","..."]}
    로 가정하고, 출력은:
      <stem>
      제약조건: A; B
    로 만든다.
    """
    if summaries is None:
        return ""

    # 이미 문자열이면 그대로 정리해서 반환
    if isinstance(summaries, str):
        return summaries.strip()

    # dict 처리
    if isinstance(summaries, dict):
        stem = (summaries.get("stem") or "").strip()
        constraints = summaries.get("constraints") or []
        # 비문자 값 방지
        constraints = [str(c).strip() for c in constraints if str(c).strip()]

        if constraints:
            return f"{stem}\n제약조건: " + "; ".join(constraints)
        else:
            return stem

    # list/기타는 줄바꿈으로 이어붙이기
    if isinstance(summaries, list):
        parts = [str(x).strip() for x in summaries if str(x).strip()]
        return "\n".join(parts)

    return str(summaries).strip()



# utils.py 내부: 채점 기준 포맷터 교체
def _format_rubric(rubric):
    """
    rubric이 dict면:
      {"detail_factor": "논증 요소 분석",
       "metric": [{"point":3,"description":"..."}, ...]}
    를 다음과 같이 출력:
      '논증 요소 분석' 에 대한 채점 기준
      3점: ...
      2점: ...
      1점: ...
      0점: ...
    * 템플릿에서 이미 "[채점 기준]" 헤더를 찍으므로 여기선 본문만 반환.
    """
    if not rubric:
        return ""

    # 리스트로 들어오는 경우(안전 처리)
    if isinstance(rubric, list):
        # 가장 핵심 하나만 쓰거나, 여러 개면 첫 항목만 사용
        rubric = rubric[0] if rubric else {}

    title = ""
    metrics = []

    if isinstance(rubric, dict):
        title = (rubric.get("detail_factor") or "").strip()
        metrics = rubric.get("metric") or []
    else:
        # 예외적인 타입은 문자열로
        return str(rubric).strip()

    # point 기준으로 내림차순 정렬 (3,2,1,0 순)
    try:
        metrics = sorted(metrics, key=lambda m: int(m.get("point", 0)), reverse=True)
    except Exception:
        pass  # 정렬 실패하면 원순서 유지

    lines = []
    if title:
        lines.append(f"'{title}' 에 대한 채점 기준")

    for m in metrics:
        try:
            p = int(m.get("point", 0))
        except Exception:
            p = m.get("point", 0)
        desc = (m.get("description") or "").strip()
        # 여러 줄 설명은 그대로 두되, 앞뒤 공백만 정리
        lines.append(f"{p}점: {desc}")

    return "\n".join(lines).strip()



def _build_prompt(subjects: str, allowed_points: List[int], summaries_str: str, rubric_str: str, response: str) -> str:
    allowed_str = "/".join(str(x) for x in allowed_points) + "점" if allowed_points else "0점"
    return PROMPT.format(
        subjects=subjects,
        allowed_str=allowed_str,
        # summaries=summaries_str,
        rubric=rubric_str,
        response=response or "",
    )

# =========================
#  메인: 학습셋 만들기
# =========================

def get_datasetM(file_name: str, args=None) -> Tuple[List[str], List[str]]:
    """
    입력:
      - merge 포맷 (권장):
        {
          "meta": {...},
          "data": [
            {
              "eid": "...",
              "qid": "...",
              "factor_id": "...",
              "detail_factor_id": "...",
              "point": "0" | 0 | ...,
              "responses": [ str, str, ... ],
              "summaries": { "stem": str, "constraints": [ ... ] }  # 묶기 코드가 넣어둔 값
              "rubric": { "detail_factor": str, "metric":[{"point":int,"description":str}, ...] }
            },
            ...
          ]
        }

      - 구(레거시) 포맷:
        [ { "point": int, "response": str }, ... ]

    출력: (texts, labels)
      - texts: 완성된 프롬프트 문자열 리스트
      - labels: "0점"/"1점"/... 등 정답 라벨 문자열 리스트
    """
    with open(file_name, "r", encoding="utf-8") as f:
        raw = json.load(f)

    texts: List[str] = []
    labels: List[str] = []

    # 과목 추정
    subjects = infer_subject_from_meta(raw, fallback_path=file_name)

    # detail_factor_id별 허용 점수 집합
    allowed_by_detail = scan_allowed_points_by_detail(raw) if isinstance(raw, dict) else {"GLOBAL": _collect_global_points_legacy(raw)}

    # 신포맷
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        for it in raw["data"]:
            did = it.get("detail_factor_id", "GLOBAL")
            allowed_points = allowed_by_detail.get(did, [])
            # gold 라벨 (항상 "N점" 형태로)
            try:
                gold = f"{int(it.get('point', 0))}점"
            except Exception:
                gold = "0점"

            # summaries / rubric 포맷팅
            summaries_str = _format_summaries(it.get("summaries"))
            rubric_str = _format_rubric(it.get("rubric"))

            # 각 response를 sample로 사용 (전체 사용)
            rs = it.get("responses", [])
            if isinstance(rs, list) and rs:
                for resp in rs:
                    prompt = _build_prompt(subjects, allowed_points, summaries_str, rubric_str, resp)
                    texts.append(prompt)
                    labels.append(gold)
            else:
                # 혹시 단일 response 키만 있을 경우 대비
                resp = it.get("response", "")
                prompt = _build_prompt(subjects, allowed_points, summaries_str, rubric_str, resp)
                texts.append(prompt)
                labels.append(gold)
        return texts, labels

    # 구포맷 (리스트)
    """if isinstance(raw, list):
        allowed_points = _collect_global_points_legacy(raw)
        for it in raw:
            try:
                gold = f"{int(it.get('point', 0))}점"
            except Exception:
                gold = "0점"
            resp = it.get("response", "")
            # 구포맷에는 summaries/rubric이 없으므로 비워서 구성
            prompt = _build_prompt(subjects, allowed_points, summaries_str="", rubric_str="[채점기준]", response=resp)
            texts.append(prompt)
            labels.append(gold)
        return texts, labels"""

    raise ValueError("지원하지 않는 JSON 구조입니다.")


# 레거시 포맷용 보조
def _collect_global_points_legacy(lst: list) -> List[int]:
    s = set()
    for it in lst:
        try:
            s.add(int(it.get("point", 0)))
        except Exception:
            s.add(0)
    return sorted(list(s))
