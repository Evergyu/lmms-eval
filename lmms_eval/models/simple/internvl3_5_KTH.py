"""internvl3_5_KTH — evaluate structurally-modified InternVL checkpoints (e.g. SFE /
LLaVA-SP vision modules) with vLLM-accelerated generation while keeping HF parity.

Why this exists
---------------
lmms-eval ships two relevant model backends:

* ``--model internvl3_5`` (HuggingFace): builds prompts from the task's
  ``doc_to_text`` + the InternVL ``conv_template`` (InternVL identity system
  message), tiles images with InternVL dynamic preprocessing, and calls
  ``model.chat(...)``. This is the *reference* path that produces the published
  scores for InternVL-derived checkpoints.
* ``--model vllm`` (generic): builds prompts from the task's ``doc_to_messages``
  with a task-specific ``SYSTEM_PROMPT``. For InternVL-derived checkpoints this is
  a *different prompt*, so scores diverge from the HF reference (we observed
  prismm_bench_identification 0.41 vs 0.52).

For a checkpoint that is "InternVL + an extra vision module, then SFT'd", only the
*inference engine* should change (HF → vLLM); the prompt / tiling / conversation
template must stay byte-for-byte identical to the HF path, otherwise you are no
longer measuring the same model.

What this wrapper does
----------------------
``InternVL3_5_KTH`` subclasses lmms-eval's ``InternVL3`` model so it inherits the
*entire* HF prompt + image pipeline unchanged. It then swaps only the LLM decode:

1. The custom vision module (SFE / LLaVA-SP) runs eagerly in PyTorch via the
   original HF model — it is a one-shot prefill, so vLLM would not accelerate it
   anyway, and running it in HF guarantees bit-exact image features.
2. The language model (Qwen3) is extracted from the checkpoint into a standalone
   vLLM model and driven through ``prompt_embeds`` (``enable_prompt_embeds``).
   ``model.chat(...)`` internally calls ``language_model.generate(inputs_embeds=...)``;
   we monkeypatch exactly that call to route through vLLM and return token ids in
   the shape the HF ``chat`` path expects.

Net effect: identical inputs to the LLM, vLLM does the autoregressive decode.
Validated to reproduce the HF reference within run-to-run noise across MC, ANLS,
OCR and document benchmarks (see docs/validation.md).

Environment
-----------
Built for an env with transformers >= 5.x and a recent vLLM (prompt_embeds +
``EmbedsPrompt`` support). Older custom ``modeling_internvl_chat.py`` checkpoints
do not load cleanly under transformers 5.x; ``_apply_tf_compat_patches`` contains
the two minimal shims required, applied before the HF model is constructed.

Gate type note: the SFE modeling reads ``GATE_TYPE`` from the environment and
falls back to ``config.spatial_gate_type`` when unset. Leave ``GATE_TYPE`` unset so
the config-trained gate is used — this matches the HF reference exactly.
"""

import json
import os
import tempfile

import torch
from tqdm import tqdm

# KTH_THINK=1 prefills the assistant turn with <think> (default off = unchanged).
_THINK = os.environ.get("KTH_THINK", "0") == "1"
# 4096: EOS 로 멈추므로 상한을 올려도 짧은 응답에는 비용이 없다. 반대로 낮으면
# </think> 를 못 닫은 채 잘려 답이 아예 안 나오고 전부 오답이 된다.
# 모델 컨텍스트는 40,960 이라 프롬프트(이미지 토큰 포함)와 합쳐도 여유가 있다.
_THINK_MAX = int(os.environ.get("KTH_THINK_MAX_TOKENS", "4096"))

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import InternVL3


def _apply_tf_compat_patches(ckpt_path: str) -> None:
    """Make an older custom ``InternVLChatModel`` load under transformers 5.x.

    Two breakages, two minimal shims:

    * ``from_pretrained`` accesses ``PreTrainedModel.all_tied_weights_keys`` (added
      in 5.x) during ``caching_allocator_warmup``. The legacy class never defines
      it, so we expose it as a settable property backed by an instance attribute.
    * The custom class overrides ``tie_weights(self)`` with no ``**kwargs``, but 5.x
      calls ``tie_weights(missing_keys=...)``. We wrap the dynamically-loaded class
      method to swallow extra args.
    """
    import transformers.modeling_utils as _mu

    if not isinstance(getattr(_mu.PreTrainedModel, "all_tied_weights_keys", None), property):
        def _get(self):
            return getattr(self, "_compat_atwk", {}) or {}

        def _set(self, value):
            object.__setattr__(self, "_compat_atwk", value)

        _mu.PreTrainedModel.all_tied_weights_keys = property(_get, _set)

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel", ckpt_path)
        if not getattr(cls, "_sfe_tie_patched", False):
            _orig_tie = cls.tie_weights
            cls.tie_weights = lambda self, *args, **kwargs: _orig_tie(self)
            cls._sfe_tie_patched = True
    except Exception:
        # Not all checkpoints ship a custom modeling file; stock loading is fine.
        pass


def _extract_qwen3_llm(ckpt_path: str, out_dir: str) -> None:
    """Write a standalone Qwen3 checkpoint containing only ``language_model.*``.

    vLLM loads this as a plain causal LM and we feed it precomputed
    ``inputs_embeds`` via ``prompt_embeds``, so the vision tower never needs to
    exist on the vLLM side.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(ckpt_path, "config.json")))
    llm_cfg = cfg["llm_config"]
    llm_cfg.setdefault("architectures", ["Qwen3ForCausalLM"])
    llm_cfg.setdefault("torch_dtype", "bfloat16")
    json.dump(llm_cfg, open(os.path.join(out_dir, "config.json"), "w"))

    import shutil

    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src = os.path.join(ckpt_path, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fn))

    index = json.load(open(os.path.join(ckpt_path, "model.safetensors.index.json")))
    prefix = "language_model."
    shard_id = 0
    new_index = {"metadata": {}, "weight_map": {}}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(os.path.join(ckpt_path, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    tensors[key[len(prefix):]] = f.get_tensor(key)
        if not tensors:
            continue
        name = f"model-{shard_id:05d}.safetensors"
        save_file(tensors, os.path.join(out_dir, name), metadata={"format": "pt"})
        for key in tensors:
            new_index["weight_map"][key] = name
        shard_id += 1
    json.dump(new_index, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))


def _checkpoint_content_key(ckpt_path: str) -> str:
    """Return a content-derived key for the extracted decoder cache.

    The former path-only key silently reused an old vLLM decoder whenever a
    conventional checkpoint directory such as ``hf_final`` was overwritten.
    """
    import hashlib

    h = hashlib.sha256()
    manifest = os.path.join(ckpt_path, "merge_manifest.json")
    if os.path.isfile(manifest):
        data = json.load(open(manifest))
        payload = {
            "format": data.get("format"), "source_shards": data.get("source_shards"),
            "output_shards": data.get("output_shards"),
            "config_sha256": data.get("config_sha256"), "index_sha256": data.get("index_sha256"),
        }
        h.update(json.dumps(payload, sort_keys=True).encode())
    else:
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path, "rb") as f:
            index_bytes = f.read()
        h.update(index_bytes)
        index = json.loads(index_bytes)
        for name in sorted(set(index["weight_map"].values())):
            stat = os.stat(os.path.join(ckpt_path, name))
            h.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


# dense_convs 커널 크기 -> 그 conv 가 만드는 로컬 토큰 수
#   16:dense_2x2(2x2격자=4), 12:overlap(겹침 4x4=16), 8:dense_4x4(16), 4:dense_8x8(8x8=64)
_KTH_KERNEL_TOKENS = {16: 4, 12: 16, 8: 16, 4: 64}


def _autodetect_local_branch(ckpt_path: str):
    """체크포인트 자체 가중치에서 학습 시 ``LOCAL_BRANCH`` 를 복원한다.

    KTH spatial 모듈의 구조(따라서 ``num_image_token``)는 생성 시 ``LOCAL_BRANCH``
    환경변수로 정해지는데 config 에 저장되지 않는다. 이 값 없이 평가하면 조용히
    엉뚱한 브랜치(center_crop)로 재조립되어 학습된 ``dense_convs`` 가중치가 UNEXPECTED
    로 버려지고 이미지 토큰 수도 어긋난다. 여기서 spatial 가중치 시그니처로 올바른 값을
    되살려 로드에 외부 상태가 필요 없게 한다.

    반환: ``(local_branch, num_image_token)``. koni spatial 모듈이 없으면(순정 InternVL)
    ``(None, None)`` — 설정할 것이 없다.
    """
    import re as _re
    import struct as _st

    want = ("dense_convs", "local_dense", "local_pool_convs", "local_satt",
            "local_pos_emb", "sfe_crop", "sfe_pool")
    shapes = {}

    def _hdr(p):
        with open(p, "rb") as f:
            n = _st.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))

    try:
        idx = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.isfile(idx):
            wm = json.load(open(idx))["weight_map"]
            for sh in set(wm[k] for k in wm if any(w in k for w in want)):
                for k, v in _hdr(os.path.join(ckpt_path, sh)).items():
                    if k != "__metadata__" and any(w in k for w in want):
                        shapes[k] = v.get("shape")
        else:
            sf = os.path.join(ckpt_path, "model.safetensors")
            if os.path.isfile(sf):
                for k, v in _hdr(sf).items():
                    if k != "__metadata__" and any(w in k for w in want):
                        shapes[k] = v.get("shape")
    except Exception as _e:
        # Header/index corruption must not silently rebuild a KTH checkpoint as a
        # vanilla model.  __init__ controls whether this is fatal through the
        # strict-mode switch (strict is the default).
        raise RuntimeError(f"could not read checkpoint weight shapes: {ckpt_path}") from _e

    keys = set(shapes)
    if not any(k.startswith(("dense_convs", "sfe_crop", "local_pool_convs", "sfe_pool"))
               for k in keys):
        return None, None  # spatial 모듈 없음 = 바닐라 InternVL

    has_crop = any("sfe_crop_convs" in k for k in keys)
    has_pool = any("local_pool_convs" in k for k in keys)
    has_satt = any(k.startswith("local_satt") for k in keys)
    has_pos = any("local_pos_emb" in k for k in keys)
    dense = sorted(k for k in keys if _re.match(r"dense_convs\.\d+\.weight$", k))
    kernels = [shapes[k][-1] for k in dense if shapes[k] and len(shapes[k]) >= 4]
    dense_tokens = sum(_KTH_KERNEL_TOKENS.get(k, 0) for k in kernels)

    if has_crop and dense:
        branch, local = "hybrid", 6 + dense_tokens          # center_crop 6 + dense
    elif has_crop:
        branch, local = "center_crop", 6
    elif has_pool:
        branch, local = "global_ca", 6
    elif dense:
        local = dense_tokens
        if has_satt:
            branch = "dense_4x4_satt"
        elif has_pos:
            branch = "dense_4x4_pos"
        elif sorted(kernels) == [8, 16]:
            branch = "dense_ms"                 # 정규 조합만(4x4+2x2=16+4=20)
        elif kernels == [16]:
            branch = "dense_2x2"
        elif kernels == [12]:
            branch = "dense_4x4_overlap"
        elif kernels == [4]:
            branch = "dense_8x8"
        elif kernels == [8]:
            branch = "dense_4x4"
        else:
            # dense_convs 가 명백히 있는데 커널 조합을 못 알아봄. 조용히 (None→center_crop)
            # 으로 떨어지면 이 패치가 없애려던 원래 버그(dense 가중치 버려짐)가 재발한다.
            # 추측 대신 크게 실패시킨다.
            raise ValueError(
                f"[KTH autodetect] dense_convs present but unrecognized kernel(s)="
                f"{kernels} in {ckpt_path}; refusing to guess LOCAL_BRANCH "
                f"(set KTH_AUTODETECT_BRANCH=0 and export LOCAL_BRANCH manually)."
            )
    else:
        branch, local = "none", 0   # poolonly (로컬 브랜치 없음)

    return branch, 256 + 6 + local   # base 256 + sfe(6 scales) + local


@register_model("internvl3_5_KTH")
class InternVL3_5_KTH(InternVL3):
    """InternVL3-family HF pipeline with vLLM-accelerated LLM decode.

    Args (beyond ``InternVL3``):
        llm_gpu_memory_utilization: vLLM ``gpu_memory_utilization`` for the Qwen3
            decoder. The HF model (vision + ViT) also lives on the same device, so
            keep headroom (0.4 is a safe default on an 80GB+ GPU).
        llm_max_model_len: vLLM ``max_model_len``. Set at or above the longest
            (text + image-token) sequence your tasks produce to avoid truncation
            relative to the HF reference.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        llm_gpu_memory_utilization: float = 0.4,
        llm_max_model_len: int = 12288,
        gen_batch: int = 1,
        **kwargs,
    ):
        # Some lmms-eval task utils import OpenAI clients at module load time.
        os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-offline")

        # gen_batch: how many single-image / text-only requests to decode together
        # through vLLM's continuous batching (max_num_seqs). 1 = original strictly
        # sequential path (bit-exact HF reproduction). >1 fills the GPU but, since
        # vLLM is not batch-invariant, may shift a few near-tie answers — re-baseline
        # at the chosen value. Multi-image requests always fall back to the 1-by-1
        # path. Env GEN_BATCH overrides the model_args value.
        self._gen_batch = max(1, int(os.environ.get("GEN_BATCH", gen_batch)))

        # Patch transformers BEFORE the HF model is constructed in super().__init__.
        _apply_tf_compat_patches(pretrained)

        # KTH spatial 브랜치 구조는 생성 시 LOCAL_BRANCH env 로 정해지는데 config 에
        # 저장되지 않는다 → 이 값 없이 평가하면 조용히 center_crop 으로 재조립되어 학습된
        # dense 가중치를 버리고 토큰 수도 어긋난다. 체크포인트 가중치 시그니처에서 되살려
        # 학습과 일치시킨다. 끄기: KTH_AUTODETECT_BRANCH=0.
        if os.environ.get("KTH_AUTODETECT_BRANCH", "1") == "1":
            try:
                _lb, _nit = _autodetect_local_branch(pretrained)
            except Exception as _e:
                if os.environ.get("KTH_AUTODETECT_STRICT", "1") == "1":
                    raise RuntimeError(f"KTH local-branch autodetection failed for {pretrained}") from _e
                print(f"[KTH autodetect] WARNING: {_e}; strict mode disabled", flush=True)
                _lb, _nit = None, None
            if _lb is not None:
                _prev = os.environ.get("LOCAL_BRANCH")
                if _prev not in (None, _lb):
                    print(f"[KTH autodetect] overriding LOCAL_BRANCH={_prev!r} -> "
                          f"{_lb!r} (from weights; num_image_token={_nit})", flush=True)
                else:
                    print(f"[KTH autodetect] LOCAL_BRANCH={_lb} "
                          f"num_image_token={_nit} (from weights)", flush=True)
                os.environ["LOCAL_BRANCH"] = _lb
            else:
                # 이 체크포인트는 koni 모듈 없음(바닐라). 같은 프로세스에서 앞선 KTH 모델이
                # 남긴 stale LOCAL_BRANCH 를 물려받지 않도록 제거(프로세스 재사용 방어).
                os.environ.pop("LOCAL_BRANCH", None)

        super().__init__(pretrained=pretrained, **kwargs)  # HF model / tokenizer / tiling, unchanged.

        # 2026-08-27: 순수 HF 경로 스위치(기본 OFF = 기존 동작 그대로).
        #   KTH_USE_HF=1 이면 vLLM 엔진을 아예 만들지 않고 monkeypatch 도 걸지 않는다.
        #   즉 model.chat() 이 원래의 language_model.generate 를 그대로 쓴다.
        #   왜 필요한가: vLLM 경로(prompt_embeds)는 async_scheduling 버그로 디코드가
        #   오염된 전력이 있다(analysis/VLLM_DECODE_DIVERGENCE.md). 그 버그를 끈 뒤에도
        #   "vLLM 과 HF 가 같은 답을 내는가" 는 별도 질문이며, 그 기준선이 이 경로다.
        #   느리다(배치 없이 HF generate). 작은 벤치·표본 대조용으로만 써라.
        if os.environ.get("KTH_USE_HF", "0") == "1":
            self._vllm = None
            self._SamplingParams = None
            print("[KTH] KTH_USE_HF=1 — vLLM 우회, 순수 HF model.chat() 경로로 디코드한다.",
                  flush=True)
            return

        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams

        # Extract the Qwen3 decoder once per checkpoint and load it into vLLM.
        # Cache the extracted decoder on DISK, never /tmp: on this host /tmp is
        # tmpfs (RAM), so dumping a ~14GB decoder per checkpoint across a sweep
        # exhausts memory. Key by checkpoint CONTENT so reusing a path cannot
        # combine a new HF vision model with a stale vLLM decoder.

        cache_root = os.environ.get("KTH_LLM_CACHE_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "internvl_kth_qwen3"
        )
        key = _checkpoint_content_key(pretrained)
        tmp_dir = os.path.join(cache_root, f"qwen3_{key}")
        ready = os.path.join(tmp_dir, "CACHE_READY.json")
        if not os.path.exists(ready):
            import shutil

            os.makedirs(cache_root, exist_ok=True)
            build_dir = tempfile.mkdtemp(prefix=f".qwen3_{key}_", dir=cache_root)
            try:
                _extract_qwen3_llm(pretrained, build_dir)
                json.dump({"checkpoint": os.path.abspath(pretrained), "content_key": key},
                          open(os.path.join(build_dir, "CACHE_READY.json"), "w"))
                try:
                    os.rename(build_dir, tmp_dir)
                except FileExistsError:
                    shutil.rmtree(build_dir)
            except Exception:
                shutil.rmtree(build_dir, ignore_errors=True)
                raise
        cache_meta = json.load(open(ready))
        if cache_meta.get("content_key") != key:
            raise RuntimeError(f"vLLM cache key mismatch: {tmp_dir}")
        self._vllm = LLM(
            model=tmp_dir,
            trust_remote_code=True,
            gpu_memory_utilization=float(llm_gpu_memory_utilization),
            max_model_len=int(llm_max_model_len),
            max_num_seqs=self._gen_batch,
            dtype="bfloat16",
            enable_prompt_embeds=True,
            enforce_eager=True,
            # 비침습 추가(2026-07-21): 기본 1=기존 동작 그대로. VLLM_TP=2 로 줄 때만
            # vLLM 디코더를 2 GPU 텐서병렬 → 38B의 HF+vLLM 이중로드 단일GPU 초과 해결.
            tensor_parallel_size=int(os.environ.get("VLLM_TP", "1")),
            # 2026-08-26: async_scheduling 을 끈다. enable_prompt_embeds=True 와 겹치면
            # 디코드 출력이 배치 뒤쪽일수록 오염된다(실측: chartqa 배치 첫 자리 105건 정크 0%,
            # 전체 32.3%, 위치 19 에서 63.5%). 원인은 stock vLLM 의
            # gpu_model_runner._prepare_input_ids — 빠른 경로에는
            # `is_token_ids.gpu[:num_common_tokens]=True` 가 있는데 scatter(재정렬) 경로에는
            # 없어서, prompt_embeds 모드에서 직전 prefill 이 남긴 이미지 토큰 임베딩이
            # "직전 생성 토큰" 자리로 들어간다. 끄면 그 분기에 도달하지 않는다.
            # 되돌리기: KTH_VLLM_ASYNC=1
            async_scheduling=(os.environ.get("KTH_VLLM_ASYNC", "0") == "1"),
        )

        tokenizer = self._tokenizer
        vllm_engine = self._vllm
        SamplingParams = self._SamplingParams

        # vLLM 은 stop token 을 안 주면 답을 낸 뒤에도 max_tokens(기본 1024)까지 계속 생성해
        # 꼬리가 오염된다('B numpy', 'C\\boxed{C}' 등 → exact-match 벤치 점수 저하). HF chat
        # 경로(generate_until, `gk["eos_token_id"]=convert_tokens_to_ids(sep)`)가 멈추는 것과
        # 동일한 대화 구분자(conv template sep) 토큰에서 vLLM 도 멈추게 한다 → HF 와 같은 지점
        # 정지라 답이 잘릴 일 없음(HF 가 그 토큰으로 정상 정지). 끄기: KTH_VLLM_STOP=0.
        _vllm_stop_ids = None
        if os.environ.get("KTH_VLLM_STOP", "1") == "1":
            try:
                import sys as _sys
                _get_ct = _sys.modules[type(self._model).__module__].get_conv_template
                _sep = _get_ct(self._model.template).sep.strip()
                _sid = tokenizer.convert_tokens_to_ids(_sep)
                if isinstance(_sid, int) and _sid >= 0:
                    _vllm_stop_ids = [_sid]                       # HF 와 동일한 sep 토큰
            except Exception as _e:
                print(f"[KTH] vLLM stop-token(sep) 감지 실패({_e})", flush=True)
            if _vllm_stop_ids is None and tokenizer.eos_token_id is not None:
                _vllm_stop_ids = [int(tokenizer.eos_token_id)]    # 폴백: 기본 eos
            print(f"[KTH] vLLM stop_token_ids={_vllm_stop_ids} (HF sep 미러, 꼬리오염 방지)",
                  flush=True)

        def _vllm_generate(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            generation_config=None,
            **kwargs,
        ):
            """Drop-in for ``language_model.generate`` using vLLM via prompt_embeds.

            ``model.chat(...)`` always supplies ``inputs_embeds`` (image tokens already
            fused), so we only support that path. Returns a ``(batch, seq)`` LongTensor
            of generated token ids, which ``chat`` decodes with the tokenizer.
            """
            assert inputs_embeds is not None, "internvl3_5_KTH supports the inputs_embeds path only"
            max_new_tokens = (
                (getattr(generation_config, "max_new_tokens", None) if generation_config is not None else None)
                or kwargs.get("max_new_tokens")
                or 1024
            )
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=int(max_new_tokens),
                stop_token_ids=_vllm_stop_ids,   # None=기존 동작(정지 없음), 위에서 HF sep 미러
            )

            # Build one prompt-embeds entry per batch row. batch_chat left-pads, so
            # drop padding positions via attention_mask before handing the real
            # prompt embeddings to vLLM (a single image-free row also passes its
            # full sequence here, since its mask is all ones).
            prompts = []
            for i in range(inputs_embeds.shape[0]):
                emb = inputs_embeds[i]
                if attention_mask is not None:
                    emb = emb[attention_mask[i].bool()]
                prompts.append({"prompt_embeds": emb.to(torch.bfloat16).contiguous()})

            # One generate() call -> vLLM continuous-batches the whole list and
            # returns outputs in input order.
            outs = vllm_engine.generate(prompts, sampling)
            sequences = [list(o.outputs[0].token_ids) for o in outs]

            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            max_len = max((len(s) for s in sequences), default=1)
            result = torch.full(
                (len(sequences), max_len), pad_id, dtype=torch.long, device=inputs_embeds.device
            )
            for i, seq in enumerate(sequences):
                if seq:
                    result[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=inputs_embeds.device)
            return result

        # Route the HF chat path's LLM decode through vLLM.
        self._model.language_model.generate = _vllm_generate

    def generate_until(self, requests):
        """Decode ``gen_batch`` requests per vLLM call through one unified path.

        Builds each prompt exactly like ``model.chat`` (same conv template and the
        same per-image ``<image>`` -> image-token substitution driven by each
        image's tile count), so single-image, multi-image and text-only requests
        all batch the same way -- no per-request fallback, one progress bar, one
        logging path. Requests are grouped by (has-image, generation kwargs) so a
        batch shares image-presence and ``max_new_tokens``.

        ``gen_batch == 1`` (default) or non-image modalities defer to the unchanged
        parent path. Pair ``gen_batch > 1`` with ``VLLM_BATCH_INVARIANT=1`` to get
        outputs identical to the sequential path regardless of batch size.
        """
        if self._gen_batch <= 1 or self.modality != "image":
            return super().generate_until(requests)

        import sys

        from lmms_eval.models.simple import internvl3 as _iv3

        model = self.model
        tok = self.tokenizer
        get_conv_template = sys.modules[type(model).__module__].get_conv_template
        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        sep = get_conv_template(model.template).sep.strip()
        model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CTX)

        def _norm_gk(gen_kwargs):
            gk = dict(gen_kwargs)
            gk.pop("until", None)
            for k, v in _iv3.DEFAULT_GEN_KWARGS.items():
                gk.setdefault(k, v)
            for k in [k for k in gk if k not in _iv3.DEFAULT_GEN_KWARGS]:
                gk.pop(k)
            if _THINK:
                # A reasoning block costs tokens before the answer starts (~430 for
                # Korean, measured). Short-answer tasks set max_new_tokens as low as 16,
                # which would truncate mid-thought and score every one of them wrong.
                # Generation still stops at EOS, so a high floor costs nothing on the
                # answers that stay short.
                gk["max_new_tokens"] = max(int(gk.get("max_new_tokens") or 0), _THINK_MAX)
            return gk

        def _build(contexts, visuals):
            """Replicate model.chat prompt building. Returns (query, pixel_values)."""
            n = len(visuals)
            processed = []
            if n == 0:
                question = contexts
            else:
                dyn = max(1, min(self.max_num, self.total_max_num // n))
                processed = [
                    _iv3.load_image(v, max_num=dyn).to(torch.bfloat16).to(self._device) for v in visuals
                ]
                # Match parent: keep author-interleaved tags, else prepend one per image.
                question = contexts if contexts.count("<image>") == n else " ".join(["<image>"] * n) + "\n" + contexts
            template = get_conv_template(model.template)
            template.system_message = model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            if _THINK:
                # Prime the assistant turn with <think> so the model opens a reasoning
                # block. <think>/</think> are single tokens here (151667 / 151668).
                # The model was not RL'd on a think contract (v9 THINK_W defaults to 0),
                # so this exercises what the base model retained — measure, don't assume.
                query += "<think>\n"
            for p in processed:  # one <image> -> one image's token block, in order
                block = IMG_START + IMG_CTX * model.num_image_token * p.size(0) + IMG_END
                query = query.replace("<image>", block, 1)
            pv = torch.cat(processed, dim=0) if processed else None
            return query, pv

        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def _run(chunk):
            # OOM 은 model.generate 뿐 아니라 pixel_values=torch.cat(...) / .to(device)
            # 에서도 난다(다중페이지 dude 는 타일이 많음). 전체를 감싸 '어디서 터지든' 잡는다.
            input_ids = attn = pixel_values = out = None
            try:
                gk = dict(chunk[0][3])
                gk["eos_token_id"] = tok.convert_tokens_to_ids(sep)
                tok.padding_side = "left"
                enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True)
                input_ids = enc["input_ids"].to(self._device)
                attn = enc["attention_mask"].to(self._device)
                pvs = [c[2] for c in chunk]
                pixel_values = torch.cat(pvs, dim=0) if pvs[0] is not None else None
                out = model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn, **gk)
                for c, r in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
                    res[c[0]] = r.split(sep)[0].strip()
                    pbar.update(1)
            except torch.OutOfMemoryError:
                # 배치를 반으로 쪼개 재귀 재시도(타일/화질/결과 동일, 스케줄만 달라짐).
                # 재귀 전에 이번 프레임의 GPU 텐서를 풀어줘야 메모리가 회수된다(프레임이 안 벗겨지므로).
                del input_ids, attn, pixel_values, out
                torch.cuda.empty_cache()
                if len(chunk) > 1:
                    mid = len(chunk) // 2
                    _run(chunk[:mid])
                    _run(chunk[mid:])
                else:
                    # batch==1 도 OOM(초대형 단일문서 or GPU 포화): 그 항목만 빈 응답으로 두고 벤치는 완주.
                    print(f"[internvl3_5_KTH] OOM on single request idx={chunk[0][0]} "
                          f"-> empty response (tiles too many or GPU saturated)", flush=True)
                    res[chunk[0][0]] = ""
                    pbar.update(1)

        # Prompt/image building (CPU-side InternVL dynamic tiling: PIL resize of up to
        # ~12 tiles per image) is the real bottleneck — it runs while the GPU sits idle.
        # Build prompts in parallel worker threads with a bounded look-ahead queue so
        # CPU preprocessing of upcoming requests overlaps GPU decode of the current
        # batch. Building is deterministic, so per-request outputs are unchanged; only
        # the scheduling differs. PREP_WORKERS / PREP_PREFETCH tune it.
        def _build_one(item):
            idx, reg = item
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args
            gk = _norm_gk(gen_kwargs)
            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            query, pv = _build(contexts, visuals)
            key = (pv is not None, tuple(sorted(gk.items())))
            return idx, query, pv, gk, key

        import itertools
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        nworkers = max(1, int(os.environ.get("PREP_WORKERS", "8")))
        prefetch = max(self._gen_batch * 2, int(os.environ.get("PREP_PREFETCH", str(self._gen_batch * 3))))

        def _built_stream(ex):
            futs = deque()
            it = iter(enumerate(requests))
            for x in itertools.islice(it, prefetch):
                futs.append(ex.submit(_build_one, x))
            for x in it:
                yield futs.popleft().result()
                futs.append(ex.submit(_build_one, x))
            while futs:
                yield futs.popleft().result()

        cur, cur_key = [], None
        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            for idx, query, pv, gk, key in _built_stream(ex):
                if cur and (key != cur_key or len(cur) >= self._gen_batch):
                    _run(cur)
                    cur = []
                cur_key = key
                cur.append((idx, query, pv, gk))
                if len(cur) >= self._gen_batch:
                    _run(cur)
                    cur = []
            if cur:
                _run(cur)
        pbar.close()
        return res

    def flatten(self, input):
        """Flatten nested visuals, skipping ``None`` entries.

        The base ``InternVL3.flatten`` does ``for i in input: for j in i``, which
        raises ``TypeError`` on text-only tasks (e.g. ``scibench``) where
        ``doc_to_visual`` returns ``None``. Guarding here lets such tasks fall
        through to the no-image (text-only) path in ``generate_until``.
        """
        new_list = []
        for i in input:
            if i is None:
                continue
            for j in i:
                new_list.append(j)
        return new_list
