# ChartQAPro task utils — upstream lmms-eval v0.7.1이 yaml만 배포하고 이 파일을 누락해서 직접 작성 (2026-06-12).
# 채점 로직은 공식 저장소 vis-nlp/ChartQAPro의 evaluate_predictions.py를 그대로 이식했고,
# 프롬프트는 저자 추천 구현(VLMEvalKit, qa_type=Direct)의 question type별 instruction을 따름.
# 데이터셋: ahmed-masry/ChartQAPro (test 1948개; yaml의 lmms-lab/ChartQAPro는 허브에 존재하지 않음)

import ast
import re
from io import BytesIO
from typing import Any, List, Optional

from anls import anls_score
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# 공식 evaluate_predictions.py 이식 (vis-nlp/ChartQAPro)

def fix_list_format(item: str) -> Any:
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> Optional[List[str]]:
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def to_float(text: str) -> Optional[float]:
    try:
        return float(text.strip().strip("%"))
    except ValueError:
        return None


def evaluate_single_answer(target: str, prediction: str, max_relative_change: float = 0.05) -> float:
    t = target.strip().strip("%").strip()
    p = prediction.strip().strip("%").strip()
    t_f = to_float(t)
    p_f = to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0
    return anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5)


def relaxed_correctness_chartqapro(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    year_flags: Optional[List[str]] = None,
    always_use_exact_match: bool = False,
) -> float:
    fixed_t = fix_list_format(target)
    t_list = parse_to_list(str(fixed_t)) or [str(target)]
    p_list = parse_to_list(str(prediction)) or [str(prediction)]
    n = len(t_list)
    if year_flags is not None and len(year_flags) < n:
        year_flags = year_flags * n

    scores: List[float] = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue
        t_item, p_item, flag = t_list[idx], p_list[idx], year_flags[idx]
        flag_cond = True if flag.upper() == "YES" else False
        if flag_cond or always_use_exact_match:
            try:
                scores.append(1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0)
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(evaluate_single_answer(t_item, p_item, max_relative_change))
    return sum(scores) / len(scores) if scores else 0.0


# ─────────────────────────────────────────────────────────────────
# lmms-eval 훅

# VLMEvalKit qa_type=Direct instruction (question type별)
_COMMON_GUIDE = (
    " If the question is not answerable from the chart, answer 'unanswerable'."
    " Do not generate units unless the chart explicitly shows them."
    " If the question has multiple answers, format them as ['Answer1', 'Answer2']."
    " Return only the final answer without any additional text."
)
_TYPE_INSTRUCTIONS = {
    "Factoid": "You are given a factoid question that you need to answer based on the provided image. Your answer should be a single word, number, or phrase." + _COMMON_GUIDE,
    "Multi Choice": "Your answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text)." + _COMMON_GUIDE,
    "Conversational": "You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image." + _COMMON_GUIDE,
    "Fact Checking": "Your answer should be either true or false (without any additional text)." + _COMMON_GUIDE,
    "Hypothetical": "You are given a hypothetical question that you need to answer based on the provided image. Your answer should be a single word, number, or phrase." + _COMMON_GUIDE,
}
_DEFAULT_INSTRUCTION = _TYPE_INSTRUCTIONS["Factoid"]


def chartqapro_doc_to_visual(doc):
    img = doc["image"]
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))
    elif isinstance(img, dict) and "bytes" in img:
        img = Image.open(BytesIO(img["bytes"]))
    return [img.convert("RGB")]


def chartqapro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    qtype = doc.get("Question Type", "")
    instruction = _TYPE_INSTRUCTIONS.get(qtype, _DEFAULT_INSTRUCTION)
    parts = [instruction]
    paragraph = doc.get("Paragraph")
    if paragraph and str(paragraph).strip() and str(paragraph).strip().lower() != "nan":
        parts.append(str(paragraph).strip())
    questions = doc.get("Question") or []
    if isinstance(questions, str):
        questions = [questions]
    parts.append("\n".join(q.strip() for q in questions))
    return "\n".join(parts)


def chartqapro_process_results(doc, results):
    pred = str(results[0]).strip().strip(".").strip("\n")
    gt = str(doc["Answer"][-1]).strip(".").strip("\n")
    qtype = doc.get("Question Type", "")
    year_flags = list(doc.get("Year") or ["NO"])
    if qtype == "Conversational":
        year_flags = year_flags[-1:]
    # 공식 스크립트는 always_use_exact_match를 계산만 하고 채점 함수에 전달하지 않는 버그가 있음.
    # VLMEvalKit과 동일하게 실제로 전달한다 (Fact Checking/Multi Choice는 한 단어라 결과 차이는 미미).
    always_exact = qtype in ("Fact Checking", "Multi Choice")
    score = relaxed_correctness_chartqapro(gt, pred, year_flags=year_flags, always_use_exact_match=always_exact)
    return {"relaxed_overall": score}
