import math

from pdf2image import convert_from_bytes
from PIL import Image

SYSTEM_PROMPT = "Answer with only the letter of the correct answer (A, B, C, or D), do not output anything else."


def concat_images(images, column_num=3):
    if not images:
        return None

    if column_num == 1:
        total_height = images[0].height * len(images)
        total_width = images[0].width
    else:
        rows = math.ceil(len(images) / column_num)
        total_height = images[0].height * rows
        total_width = images[0].width * column_num

    concatenated_image = Image.new("RGB", (total_width, total_height), "white")

    x_offset, y_offset = 0, 0
    for cnt, image in enumerate(images):
        concatenated_image.paste(image, (x_offset, y_offset))
        x_offset += image.width
        if (cnt + 1) % column_num == 0:
            y_offset += image.height
            x_offset = 0

    return concatenated_image


import os as _os

# HF 데이터셋(daluggas/PRISMM-Bench)은 384행 중 180행의 pdf가 비어 있음.
# 논문은 384행 전체를 page/document context로 평가하므로, 빠진 PDF는
# OpenReview(https://openreview.net/pdf?id=<paper_id>)에서 받아 여기 캐시에 둠.
# 채우는 방법: uid 앞부분(paper_id)으로 pdf_cache/<paper_id>.pdf 저장.
PDF_CACHE_DIR = _os.environ.get("PRISMM_PDF_DIR", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pdf_cache"))


def _local_pdf_path(doc):
    uid = doc.get("uid", "")
    paper_id = uid.rsplit("_", 1)[0] if uid else ""
    return _os.path.join(PDF_CACHE_DIR, f"{paper_id}.pdf") if paper_id else ""


def get_pdf_bytes(doc):
    pdf_data = doc.get("pdf")
    if pdf_data is not None:
        return pdf_data.stream.getvalue()
    local = _local_pdf_path(doc)
    if local and _os.path.isfile(local):
        with open(local, "rb") as f:
            return f.read()
    raise ValueError("PDF data not found in document")


def prismm_doc_to_visual(doc):
    """Extract all images from the 'parts' field of the document."""
    visuals = []
    parts = doc.get("parts", {})

    if isinstance(parts, dict):
        content_images = parts.get("content_image", [])
        for img in content_images:
            if img is not None:
                visuals.append(img.convert("RGB"))

    return visuals


def prismm_doc_to_visual_whole_page(doc):
    visuals = []
    pdf_bytes = get_pdf_bytes(doc)
    if not pdf_bytes:
        return visuals

    parts = doc.get("parts", {})

    page_nums: list[int] = parts.get("page", [])
    for page_num in page_nums:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=144, first_page=page_num, last_page=page_num)
            if images:
                visuals.append(images[0].convert("RGB"))
        except Exception as e:
            pass
    return visuals


def prismm_doc_to_visual_whole_doc(doc):
    visuals = []
    pdf_bytes = get_pdf_bytes(doc)
    if not pdf_bytes:
        return visuals

    try:
        images = convert_from_bytes(pdf_bytes, dpi=144, last_page=1000)  # safeguard for very large documents
        max_images = 5
        column_num = 3

        if not images:
            return visuals

        pages_per_image = math.ceil(len(images) / max_images)

        for i in range(0, len(images), pages_per_image):
            batch_pages = images[i : i + pages_per_image]
            concatenated_image = concat_images(batch_pages, column_num)
            if concatenated_image:
                visuals.append(concatenated_image.convert("RGB"))
            if len(visuals) >= max_images:
                break
    except Exception:
        pass

    return visuals


def prismm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct the question text with all text parts and multiple choice options."""
    task_identification = doc.get("task_identification", {})
    question = task_identification.get("question", "What is the inconsistency in these parts of a scientific paper?")
    choices = task_identification.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])

        for i, text in enumerate(content_texts):
            if text:
                text_parts.append(text)

    context = "\n\n".join(text_parts) if text_parts else ""

    choices_str = "\n".join(choices)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {
            "pre_prompt": "",
            "post_prompt": "\nOutput only a single letter corresponding to the correct answer. Do not output any explanation or text of the question again.",
        }

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nOutput only a single letter corresponding to the correct answer. Do not output any explanation or text of the question again.",
    )
    post_prompt = _os.environ.get("PRISMM_POST_PROMPT", post_prompt)  # 추론프롬프트 실험 오버라이드(미설정시 기존동작)

    if context:
        full_text = f"{pre_prompt}\n{context}\n\n{question}\n{choices_str}{post_prompt}"
    else:
        full_text = f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    return full_text


def _prismm_doc_to_messages_generic(doc, lmms_eval_specific_kwargs, visual_fn, skip_text_context=False):
    task_identification = doc.get("task_identification", {})
    question = task_identification.get("question", "What is the inconsistency in these parts of a scientific paper?")
    choices = task_identification.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []
    if isinstance(parts, dict) and not skip_text_context:
        content_texts = parts.get("content_text", [])
        for i, text in enumerate(content_texts):
            if text:
                text_parts.append(text)
    context = "\n\n".join(text_parts) if text_parts else ""

    letters = task_identification.get("letters", [])
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    letter_choice_pairs = list(zip(letters, choices))
    letter_choice_pairs.sort(key=lambda x: x[0])

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt = _os.environ.get("PRISMM_POST_PROMPT", post_prompt)  # 추론프롬프트 실험 오버라이드(미설정시 기존동작)

    visuals = visual_fn(doc)

    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]

    for img in visuals:
        messages[0]["content"].append({"type": "image", "url": img})

    if context:
        messages[0]["content"].append({"type": "text", "text": context})

    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    messages[0]["content"].append({"type": "text", "text": question})

    for letter, choice in letter_choice_pairs:
        messages[0]["content"].append({"type": "text", "text": choice})

    if post_prompt:
        messages[0]["content"].append({"type": "text", "text": post_prompt})

    messages = system_messages + messages
    return messages


def prismm_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return _prismm_doc_to_messages_generic(doc, lmms_eval_specific_kwargs, prismm_doc_to_visual)


def prismm_doc_to_messages_whole_page(doc, lmms_eval_specific_kwargs=None):
    return _prismm_doc_to_messages_generic(
        doc,
        lmms_eval_specific_kwargs,
        prismm_doc_to_visual_whole_page,
        skip_text_context=True,
    )


def prismm_doc_to_messages_whole_doc(doc, lmms_eval_specific_kwargs=None):
    return _prismm_doc_to_messages_generic(
        doc,
        lmms_eval_specific_kwargs,
        prismm_doc_to_visual_whole_doc,
        skip_text_context=True,
    )


def prismm_process_results(doc, results):
    """Process the model results and compare with the ground truth answer."""
    pred = results[0].strip()
    task_identification = doc.get("task_identification", {})
    target = task_identification.get("answer", "").strip().upper()

    pred_upper = pred.upper()

    if pred_upper == target:
        return {"exact_match": 1.0}

    if len(pred_upper) >= 1 and pred_upper[0] == target:
        return {"exact_match": 1.0}

    return {"exact_match": 0.0}


def prismm_edit_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct the question text for the task_remedy task."""
    task_remedy = doc.get("task_remedy", {})
    question = task_remedy.get(
        "question",
        "What action needs to be taken to resolve the inconsistency in these parts of a scientific paper?",
    )
    choices = task_remedy.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []

    if isinstance(parts, dict):
        content_texts = parts.get("content_text", [])

        for i, text in enumerate(content_texts):
            if text:
                text_parts.append(text)

    context = "\n\n".join(text_parts) if text_parts else ""

    choices_str = "\n".join(choices)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt = _os.environ.get("PRISMM_POST_PROMPT", post_prompt)  # 추론프롬프트 실험 오버라이드(remedy)

    if context:
        full_text = f"{pre_prompt}\n{context}\n\n{question}\n{choices_str}{post_prompt}"
    else:
        full_text = f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    return full_text


def prismm_edit_doc_to_target(doc):
    """Get the target answer for task_remedy."""
    task_remedy = doc.get("task_remedy", {})
    return task_remedy.get("answer", "")


def _prismm_edit_doc_to_messages_generic(doc, lmms_eval_specific_kwargs, visual_fn, skip_text_context=False):
    task_remedy = doc.get("task_remedy", {})
    question = task_remedy.get(
        "question",
        "What action needs to be taken to resolve the inconsistency in these parts of a scientific paper?",
    )
    choices = task_remedy.get("choices", [])

    parts = doc.get("parts", {})
    text_parts = []
    if isinstance(parts, dict) and not skip_text_context:
        content_texts = parts.get("content_text", [])
        for i, text in enumerate(content_texts):
            if text:
                text_parts.append(text)
    context = "\n\n".join(text_parts) if text_parts else ""

    letters = task_remedy.get("letters", [])
    if not letters or len(letters) != len(choices):
        letters = [chr(ord("A") + i) for i in range(len(choices))]

    letter_choice_pairs = list(zip(letters, choices))
    letter_choice_pairs.sort(key=lambda x: x[0])

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt = _os.environ.get("PRISMM_POST_PROMPT", post_prompt)  # 추론프롬프트 실험 오버라이드(미설정시 기존동작)

    visuals = visual_fn(doc)

    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]

    for img in visuals:
        messages[0]["content"].append({"type": "image", "url": img})

    if context:
        messages[0]["content"].append({"type": "text", "text": context})

    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    messages[0]["content"].append({"type": "text", "text": question})

    for letter, choice in letter_choice_pairs:
        messages[0]["content"].append({"type": "text", "text": f"{letter}) {choice}"})

    if post_prompt:
        messages[0]["content"].append({"type": "text", "text": post_prompt})

    messages = system_messages + messages
    return messages


def prismm_edit_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return _prismm_edit_doc_to_messages_generic(doc, lmms_eval_specific_kwargs, prismm_doc_to_visual)


def prismm_edit_doc_to_messages_whole_page(doc, lmms_eval_specific_kwargs=None):
    return _prismm_edit_doc_to_messages_generic(
        doc,
        lmms_eval_specific_kwargs,
        prismm_doc_to_visual_whole_page,
        skip_text_context=True,
    )


def prismm_edit_doc_to_messages_whole_doc(doc, lmms_eval_specific_kwargs=None):
    return _prismm_edit_doc_to_messages_generic(
        doc,
        lmms_eval_specific_kwargs,
        prismm_doc_to_visual_whole_doc,
        skip_text_context=True,
    )


def prismm_edit_process_results(doc, results):
    """Process the model results for task_remedy."""
    pred = results[0].strip()
    task_remedy = doc.get("task_remedy", {})
    target = task_remedy.get("answer", "").strip().upper()

    pred_upper = pred.upper()

    if pred_upper == target:
        return {"exact_match": 1.0}

    if len(pred_upper) >= 1 and pred_upper[0] == target:
        return {"exact_match": 1.0}

    return {"exact_match": 0.0}


def prismm_pair_match_filter_docs(dataset):
    """Filter the dataset to only include samples where pair_match task is available."""
    return dataset.filter(lambda x: x["task_pair_match"]["is_available"])


def prismm_pair_match_doc_to_visual(doc):
    """Extract visuals for the pair_match task."""
    visuals = []
    task = doc.get("task_pair_match", {})

    if task.get("query_type") == "image" and task.get("query_image") is not None:
        visuals.append(task["query_image"].convert("RGB"))

    candidates = task.get("candidates", [])
    for candidate in candidates:
        if candidate is not None:
            visuals.append(candidate.convert("RGB"))

    return visuals


def prismm_pair_match_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Construct the question text for the pair_match task."""
    task = doc.get("task_pair_match", {})

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt = _os.environ.get("PRISMM_POST_PROMPT", post_prompt)  # 추론프롬프트 실험 오버라이드(pair_match text)

    intro = "You are provided with a part of a scientific paper:"
    if task.get("query_type") == "text":
        query = task.get("query_text", "")
    else:
        query = "<image>"

    question = "The combination with one of the other parts within the same paper results in an inconsistency. Pick the letter of the correct answer option."

    letters = task.get("letters", [])
    if not letters:
        num_candidates = len(task.get("candidates", []))
        letters = [chr(ord("A") + i) for i in range(num_candidates)]

    options = "\n".join([f"{letter}) <image>" for letter in letters])

    return f"{pre_prompt}{intro}\n{query}\n{question}\n{options}{post_prompt}"


def prismm_pair_match_doc_to_target(doc):
    """Get the target answer for task_pair_match."""
    task = doc.get("task_pair_match", {})
    return task.get("answer", "")


def prismm_pair_match_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert document to message format for task_pair_match with proper interleaving."""
    task = doc.get("task_pair_match", {})

    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt = _os.environ.get("PRISMM_POST_PROMPT", post_prompt)  # 추론프롬프트 실험 오버라이드(pair_match msg)

    if pre_prompt:
        messages[0]["content"].append({"type": "text", "text": pre_prompt})

    messages[0]["content"].append(
        {
            "type": "text",
            "text": "You are provided with a part of a scientific paper:\n",
        }
    )

    if task.get("query_type") == "text":
        messages[0]["content"].append({"type": "text", "text": task.get("query_text", "")})
    else:
        if task.get("query_image") is not None:
            messages[0]["content"].append({"type": "image", "url": task["query_image"].convert("RGB")})

    messages[0]["content"].append(
        {
            "type": "text",
            "text": "\nThe combination with one of the other parts within the same paper results in an inconsistency. Pick the letter of the correct answer option.\n",
        }
    )

    letters = task.get("letters", [])
    candidates = task.get("candidates", [])

    if not letters:
        letters = [chr(ord("A") + i) for i in range(len(candidates))]

    for letter, candidate in zip(letters, candidates):
        messages[0]["content"].append({"type": "text", "text": f"{letter}) "})
        if candidate is not None:
            messages[0]["content"].append({"type": "image", "url": candidate.convert("RGB")})
        messages[0]["content"].append({"type": "text", "text": "\n"})

    messages[0]["content"].append({"type": "text", "text": post_prompt})

    messages = system_messages + messages
    return messages


def prismm_pair_match_process_results(doc, results):
    """Process the model results for task_pair_match."""
    pred = results[0].strip()
    task = doc.get("task_pair_match", {})
    target = task.get("answer", "").strip("'\" )").upper()

    pred_upper = pred.upper()

    if pred_upper == target:
        return {"exact_match": 1.0}

    if len(pred_upper) >= 1 and pred_upper[0] == target:
        return {"exact_match": 1.0}

    return {"exact_match": 0.0}


def prismm_filter_has_pdf(dataset):
    """whole_doc/whole_page 변형용: PDF를 구할 수 있는 행만 남김.

    데이터셋에 pdf가 있거나(204행) OpenReview에서 받아둔 로컬 캐시(pdf_cache/)에
    있으면 유지. 캐시가 다 채워져 있으면 384행 전체가 유지되어 논문 설정과 일치.
    어느 쪽에도 없는 행만 제외 (안 하면 get_pdf_bytes ValueError로 task 전체가 죽음).
    decode=False 캐스팅으로 PDF 디코딩 없이 빠르게 스캔.
    """
    from datasets.features.pdf import Pdf
    from loguru import logger

    undecoded = dataset.cast_column("pdf", Pdf(decode=False))
    keep = []
    for i, row in enumerate(undecoded):
        in_ds = row["pdf"] is not None and (row["pdf"].get("bytes") is not None or row["pdf"].get("path"))
        local = _local_pdf_path(row)
        if in_ds or (local and _os.path.isfile(local)):
            keep.append(i)
    dropped = len(undecoded) - len(keep)
    if dropped:
        logger.warning(f"prismm_bench whole_*: PDF를 구할 수 없는 {dropped}행 제외 ({len(keep)}/{len(undecoded)}행 평가). pdf_cache를 채우면 전체 평가 가능: {PDF_CACHE_DIR}")
    return dataset.select(keep)
