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
        **kwargs,
    ):
        # Some lmms-eval task utils import OpenAI clients at module load time.
        os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-offline")

        # Patch transformers BEFORE the HF model is constructed in super().__init__.
        _apply_tf_compat_patches(pretrained)
        super().__init__(pretrained=pretrained, **kwargs)  # HF model / tokenizer / tiling, unchanged.

        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams

        # Extract the Qwen3 decoder once per checkpoint and load it into vLLM.
        tmp_dir = os.path.join(tempfile.gettempdir(), "internvl_kth_qwen3_" + str(abs(hash(pretrained)) % 10**8))
        if not os.path.exists(os.path.join(tmp_dir, "model.safetensors.index.json")):
            _extract_qwen3_llm(pretrained, tmp_dir)
        self._vllm = LLM(
            model=tmp_dir,
            trust_remote_code=True,
            gpu_memory_utilization=float(llm_gpu_memory_utilization),
            max_model_len=int(llm_max_model_len),
            max_num_seqs=1,
            dtype="bfloat16",
            enable_prompt_embeds=True,
            enforce_eager=True,
        )

        tokenizer = self._tokenizer
        vllm_engine = self._vllm
        SamplingParams = self._SamplingParams

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
            sampling = SamplingParams(temperature=0.0, max_tokens=int(max_new_tokens))

            sequences = []
            for i in range(inputs_embeds.shape[0]):
                emb = inputs_embeds[i].to(torch.bfloat16).contiguous()
                out = vllm_engine.generate({"prompt_embeds": emb}, sampling)
                sequences.append(list(out[0].outputs[0].token_ids))

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
