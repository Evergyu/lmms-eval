"""
KRETA (KoTextVQA) lmms-eval task — KRETA-1=System1, KRETA-2=System2.
프롬프트/파싱/채점은 KRETA 원본(eval/infer/infer_new_vllm.py construct_prompt +
eval/evaluate.py parse_multi_choice_response/get_multi_choice_info)을 그대로 복제해
기존 KRETA 점수와 동일 기준이 되도록 한다. 이미지는 b64 문자열이라 디코드해서 PIL로 넘김.
internvl3_5_KTH 래퍼로 돌리면 KTH spatial-gate가 살아있는 충실 추론 + raw 트레이스(LMMS_LOG_FULL)가 됨.
"""
import base64
import io
import random

import numpy as np
from PIL import Image

# ── KRETA 원본과 동일한 프롬프트(construct_prompt, setting='default') ──
_DEFAULT_TAIL = (
    "Please select the correct answer from the options above. "
    "The last line of your response should be of the following format: "
    "'Answer: LETTER' (without quotes) where LETTER is one of options."
)


def _decode_image(b64_text):
    """KRETA infer의 decode_b64_image 와 동일."""
    if isinstance(b64_text, Image.Image):
        return b64_text.convert("RGB")
    b64_text = str(b64_text).strip()
    if b64_text.startswith("data:image"):
        b64_text = b64_text.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_text, validate=False)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def get_multi_choice_info(options):
    """KRETA evaluate.py 와 동일."""
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        safe_option = "" if option is None else str(option)
        index2ans[chr(ord(start_chr) + i)] = safe_option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """KRETA evaluate.py parse_multi_choice_response 그대로 복제."""
    if not isinstance(response, str):
        response = ""

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "
    else:
        response = ""

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:  # A B C D
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:  # A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if isinstance(ans, str) and ans and ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]
    return pred_index


# ── lmms-eval 인터페이스 ──
def kreta_doc_to_visual(doc):
    return [_decode_image(doc["image"])]


def kreta_doc_to_text(doc):
    option_letters = ["A", "B", "C", "D"]
    parsed_options = "\n".join([f"{l}. {doc.get(l, '')}" for l in option_letters])
    return f"Question: {doc['question']}\nOptions:\n{parsed_options}\n{_DEFAULT_TAIL}"


def kreta_process_results(doc, result):
    """단일 task 'kreta' — 한 레코드를 3개 메트릭(전체/System1/System2)에 모두 넣고,
    각 aggregation이 topic_difficulty(KRETA-1=System1, KRETA-2=System2)로 골라 채점."""
    pred = result[0] if isinstance(result, (list, tuple)) else result
    if pred is None or isinstance(pred, dict):
        pred = ""
    options = [doc.get("A"), doc.get("B"), doc.get("C"), doc.get("D")]
    index2ans, all_choices = get_multi_choice_info(options)
    parsed_pred = parse_multi_choice_response(str(pred), all_choices, index2ans)
    rec = {"id": doc.get("id"), "category": doc.get("category"),
           "topic_difficulty": doc.get("topic_difficulty"),
           "pred": parsed_pred, "answer": doc["answer"],
           "correct": 1.0 if parsed_pred == doc["answer"] else 0.0}
    return {"kreta_acc": rec, "kreta_system1": rec, "kreta_system2": rec}


def _acc(results):
    return sum(r["correct"] for r in results) / len(results) if results else 0.0


def kreta_agg_all(results):
    return _acc(results)


def kreta_agg_system1(results):  # KRETA-1
    return _acc([r for r in results if r.get("topic_difficulty") == "System1"])


def kreta_agg_system2(results):  # KRETA-2
    return _acc([r for r in results if r.get("topic_difficulty") == "System2"])
