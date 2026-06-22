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
        super().__init__(pretrained=pretrained, **kwargs)  # HF model / tokenizer / tiling, unchanged.

        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams

        # Extract the Qwen3 decoder once per checkpoint and load it into vLLM.
        # Cache the extracted decoder on DISK, never /tmp: on this host /tmp is
        # tmpfs (RAM), so dumping a ~14GB decoder per checkpoint across a sweep
        # exhausts memory. Key by absolute checkpoint path with a stable digest
        # (builtin hash() is salted per process -> would re-extract every run and
        # pile up). Reused across runs; override location with KTH_LLM_CACHE_DIR.
        import hashlib

        cache_root = os.environ.get("KTH_LLM_CACHE_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "internvl_kth_qwen3"
        )
        key = hashlib.md5(os.path.abspath(pretrained).encode()).hexdigest()[:12]
        tmp_dir = os.path.join(cache_root, f"qwen3_{key}")
        if not os.path.exists(os.path.join(tmp_dir, "model.safetensors.index.json")):
            _extract_qwen3_llm(pretrained, tmp_dir)
        self._vllm = LLM(
            model=tmp_dir,
            trust_remote_code=True,
            gpu_memory_utilization=float(llm_gpu_memory_utilization),
            max_model_len=int(llm_max_model_len),
            max_num_seqs=self._gen_batch,
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
        """Batch single-image / text requests through ``model.batch_chat`` so vLLM
        decodes ``gen_batch`` sequences at once. With ``gen_batch == 1`` (default)
        or non-image modalities this defers entirely to the unchanged parent path.

        Only requests with exactly one image and at most one ``<image>`` tag are
        grouped; multi-image / interleaved / mismatched-tag requests go one at a
        time through the parent ``generate_until`` (``batch_chat`` only substitutes
        a single image group per question). Requests are grouped by identical
        generation kwargs so a batch shares ``max_new_tokens``.
        """
        if self._gen_batch <= 1 or self.modality != "image":
            return super().generate_until(requests)

        from lmms_eval.models.simple import internvl3 as _iv3

        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        pending = []          # (idx, pixel_values_i, num_patches_i, question)
        pending_key = None    # generation-kwargs signature shared by the group

        def _flush():
            nonlocal pending, pending_key
            if not pending:
                return
            idxs = [p[0] for p in pending]
            pixel_values = torch.cat([p[1] for p in pending], dim=0)
            num_patches_list = [p[2] for p in pending]
            questions = [p[3] for p in pending]
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                questions,
                dict(pending_key),
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
            for i, r in zip(idxs, responses):
                res[i] = r
                pbar.update(1)
            pending = []
            pending_key = None

        for idx, reg in enumerate(requests):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args
            gen_kwargs = dict(gen_kwargs)
            gen_kwargs.pop("until", None)
            for k, v in _iv3.DEFAULT_GEN_KWARGS.items():
                gen_kwargs.setdefault(k, v)
            for k in [k for k in gen_kwargs if k not in _iv3.DEFAULT_GEN_KWARGS]:
                gen_kwargs.pop(k)

            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            image_num = len(visuals)

            # batch_chat substitutes one image group per question, so only single
            # image + at most one tag is safe; everything else uses the parent path.
            if image_num != 1 or contexts.count("<image>") > 1:
                _flush()
                res[idx] = super().generate_until([reg])[0]
                pbar.update(1)
                continue

            dynamic_max_num = max(1, min(self.max_num, self.total_max_num // image_num))
            pv = torch.cat(
                [_iv3.load_image(v, max_num=dynamic_max_num).to(torch.bfloat16).to(self._device) for v in visuals],
                dim=0,
            )

            key = tuple(sorted(gen_kwargs.items()))
            if pending and (key != pending_key or len(pending) >= self._gen_batch):
                _flush()
            pending_key = key
            pending.append((idx, pv, pv.size(0), contexts))
            if len(pending) >= self._gen_batch:
                _flush()

        _flush()
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
