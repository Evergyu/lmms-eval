import ast
import math
import re
from functools import lru_cache
from typing import Any

from huggingface_hub import hf_hub_download
from loguru import logger as eval_logger

from lmms_eval.api.metrics import levenshtein_distance

try:
    import fitz as _fitz  # PyMuPDF — self-contained, no system deps
except ImportError:
    _fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

_DATASET_REPO_ID = "yubo2333/MMLongBench-Doc"
_UNANSWERABLE = "not answerable"
_WARNED_MISSING_PDF_RENDERER = False
_PAGE_CACHE: dict[tuple[str, int], Any] = {}
_HAS_PDF_RENDERER = _fitz is not None or convert_from_path is not None


def _safe_literal_eval(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _clean_string(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"^['\"]|['\"]$", "", text).strip()
    text = text.lstrip("$").strip()
    text = text.rstrip("%").strip()

    for suffix in [" miles", " mile", " million"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()

    return " ".join(text.split())


def _is_exact_match_candidate(text: str) -> bool:
    if "https://" in text or "http://" in text:
        return True
    if text.endswith(".py") or text.endswith(".ipynb"):
        return True
    if text.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", text):
        return True
    if "a.m." in text or "p.m." in text:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        return True
    return False


def _anls_score(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    if not groundtruth and not prediction:
        return 1.0
    if not groundtruth or not prediction:
        return 0.0

    distance = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth), len(prediction))
    anls = 1.0 - (float(distance) / float(length))
    if anls <= threshold:
        return 0.0
    return anls


def _extract_first_number(value: Any) -> float | None:
    text = str(value or "")
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _float_precision(value: float) -> int:
    text = str(value)
    if "." not in text:
        return 3
    return len(text.split(".", 1)[1])


def _is_float_equal(reference: float, prediction: float, include_percentage: bool = False, is_close: bool = False) -> bool:
    candidates = [reference]
    if include_percentage:
        candidates = [reference / 100.0, reference, reference * 100.0]

    for candidate in candidates:
        if is_close and math.isclose(candidate, prediction, rel_tol=0.01):
            return True
        precision = max(min(_float_precision(prediction), _float_precision(candidate)), 2)
        if round(prediction, precision) == round(candidate, precision):
            return True
    return False


def _parse_list(value: Any) -> list[str]:
    parsed = _safe_literal_eval(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if parsed is None:
        return []
    return [str(parsed)]


def _normalize_answer_format(value: Any) -> str:
    format_name = str(value or "Str").strip().lower()
    mapping = {
        "int": "int",
        "integer": "int",
        "float": "float",
        "str": "str",
        "string": "str",
        "list": "list",
        "none": "none",
    }
    return mapping.get(format_name, "str")


def _score_int(groundtruth: Any, prediction: Any) -> float:
    gt_num = _extract_first_number(groundtruth)
    pred_num = _extract_first_number(prediction)
    if gt_num is None or pred_num is None:
        return 0.0
    # 2026-08-26: 모델이 자릿수가 폭주한 숫자를 뱉으면 float 로 inf/nan 이 되고
    # int() 가 OverflowError 로 평가 전체를 죽인다(KTH baseline 에서 실측).
    # 유한하지 않은 예측은 그냥 오답이다 - 채점기가 죽을 일이 아니다.
    if not (math.isfinite(gt_num) and math.isfinite(pred_num)):
        return 0.0
    return float(int(gt_num) == int(pred_num))


def _score_float(groundtruth: Any, prediction: Any) -> float:
    gt_num = _extract_first_number(groundtruth)
    pred_num = _extract_first_number(prediction)
    if gt_num is None or pred_num is None:
        return 0.0
    return float(_is_float_equal(gt_num, pred_num, include_percentage=True, is_close=True))


def _score_str(groundtruth: Any, prediction: Any) -> float:
    gt_text = _clean_string(groundtruth)
    pred_text = _clean_string(prediction)

    if _is_exact_match_candidate(gt_text):
        return float(gt_text == pred_text)
    return _anls_score(gt_text, pred_text)


def _score_list(groundtruth: Any, prediction: Any) -> float:
    gt_items = sorted(_clean_string(item) for item in _parse_list(groundtruth))
    pred_items = sorted(_clean_string(item) for item in _parse_list(prediction))

    if len(gt_items) != len(pred_items):
        return 0.0
    if not gt_items:
        return 1.0

    if _extract_first_number(gt_items[0]) is not None or _is_exact_match_candidate(gt_items[0]):
        return float(gt_items == pred_items)

    return min(_anls_score(gt_item, pred_item) for gt_item, pred_item in zip(gt_items, pred_items))


def _score_answer(groundtruth: Any, prediction: Any, answer_format: Any) -> float:
    normalized_format = _normalize_answer_format(answer_format)
    if normalized_format == "int":
        return _score_int(groundtruth, prediction)
    if normalized_format == "float":
        return _score_float(groundtruth, prediction)
    if normalized_format == "list":
        return _score_list(groundtruth, prediction)
    if normalized_format == "none":
        return float(_clean_string(prediction) == _UNANSWERABLE)
    return _score_str(groundtruth, prediction)


def _is_answerable(value: Any) -> bool:
    return _clean_string(value) != _UNANSWERABLE


@lru_cache(maxsize=512)
def _download_pdf(doc_id: str) -> str:
    return hf_hub_download(
        repo_id=_DATASET_REPO_ID,
        repo_type="dataset",
        filename=f"documents/{doc_id}",
    )


def _cap_page_px(img):
    """MMLB_MAX_PAGE_PX env 설정 시 긴 변을 그 픽셀로 다운스케일 (미설정=기존 동작 그대로).

    evidence 페이지가 크면 vLLM InternVL 동적타일이 페이지당 12타일까지 붙어
    일부 문항에서 프롬프트가 max_model_len(40960)을 초과해 런 전체가 죽는다.
    기본 동작(공용 public_eval 프로토콜)은 불변; 필요 시 env 로만 제한."""
    import os as _os

    cap = _os.environ.get("MMLB_MAX_PAGE_PX")
    if not cap or img is None:
        return img
    cap = int(cap)
    if max(img.size) <= cap:
        return img
    w, h = img.size
    if w >= h:
        return img.resize((cap, max(1, int(h * cap / w))))
    return img.resize((max(1, int(w * cap / h)), cap))


def _render_page_fitz(pdf_path: str, page_number: int):
    """Render a PDF page using PyMuPDF (no system deps)."""
    from PIL import Image

    doc = _fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        doc.close()
        return None
    page = doc[page_number - 1]  # 0-indexed
    pix = page.get_pixmap(dpi=144)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return _cap_page_px(img)


def _render_page_pdf2image(pdf_path: str, page_number: int):
    """Render a PDF page using pdf2image (needs poppler)."""
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=144)
    if not images:
        return None
    return _cap_page_px(images[0].convert("RGB"))


def _render_page(doc_id: str, page_number: int):
    if not _HAS_PDF_RENDERER:
        return None

    cache_key = (doc_id, page_number)
    cached = _PAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    try:
        pdf_path = _download_pdf(doc_id)
        if _fitz is not None:
            page_image = _render_page_fitz(pdf_path, page_number)
        else:
            page_image = _render_page_pdf2image(pdf_path, page_number)
        if page_image is not None:
            _PAGE_CACHE[cache_key] = page_image
            return page_image.copy()
        return None
    except Exception as exc:
        eval_logger.warning("Failed to render {} page {}: {}", doc_id, page_number, exc)
        return None


def _parse_page_numbers(value: Any) -> list[int]:
    pages = _safe_literal_eval(value)
    if not isinstance(pages, list):
        return []

    parsed_pages = []
    for page in pages:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            parsed_pages.append(page_number)
    return parsed_pages


def mmlongbench_doc_to_visual(doc):
    global _WARNED_MISSING_PDF_RENDERER

    doc_id = str(doc.get("doc_id", "")).strip()
    if doc_id == "":
        return []

    pages = _parse_page_numbers(doc.get("evidence_pages"))
    if not pages:
        return []

    # MMLB_MAX_PAGES env 설정 시 evidence 페이지 수 상한 (균등 서브샘플, 미설정=기존 동작).
    # 일부 극단 문항(evidence 13페이지+)이 40960 컨텍스트를 넘겨 vLLM 런 전체가 죽는 것 방지.
    import os as _os

    _cap = _os.environ.get("MMLB_MAX_PAGES")
    if _cap and len(pages) > int(_cap):
        _n = int(_cap)
        _idx = sorted({round(i * (len(pages) - 1) / (_n - 1)) for i in range(_n)})
        pages = [pages[i] for i in _idx]

    if not _HAS_PDF_RENDERER:
        if not _WARNED_MISSING_PDF_RENDERER:
            eval_logger.warning("Neither PyMuPDF nor pdf2image is installed. MMLongBench-Doc will run with text-only prompts. Install with: pip install pymupdf")
            _WARNED_MISSING_PDF_RENDERER = True
        return []

    visuals = []
    seen_pages = set()
    for page_number in pages:
        if page_number in seen_pages:
            continue
        seen_pages.add(page_number)

        image = _render_page(doc_id, page_number)
        if image is not None:
            visuals.append(image)
    return visuals


def mmlongbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def mmlongbench_doc_to_target(doc):
    return str(doc.get("answer", ""))


def mmlongbench_doc_process_results(doc, results):
    prediction = str(results[0]).strip()
    answer = doc.get("answer", "")
    answer_format = doc.get("answer_format", "Str")
    score = _score_answer(answer, prediction, answer_format)

    return {
        "mmlongbench_doc_acc": {"score": score},
        "mmlongbench_doc_f1": {
            "score": score,
            "gt_answerable": _is_answerable(answer),
            "pred_answerable": _is_answerable(prediction),
        },
    }


def mmlongbench_doc_aggregate_acc(results):
    if not results:
        return 0.0
    return sum(float(item.get("score", 0.0)) for item in results) / len(results)


def mmlongbench_doc_aggregate_f1(results):
    if not results:
        return 0.0

    gt_positive = [item for item in results if item.get("gt_answerable", False)]
    pred_positive = [item for item in results if item.get("pred_answerable", False)]

    recall = sum(float(item.get("score", 0.0)) for item in gt_positive) / len(gt_positive) if gt_positive else 0.0
    precision = sum(float(item.get("score", 0.0)) for item in pred_positive) / len(pred_positive) if pred_positive else 0.0

    if recall + precision == 0.0:
        return 0.0
    return (2.0 * recall * precision) / (recall + precision)
