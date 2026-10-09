"""Hardware-adaptive loader for Qwen3-TTS VoiceDesign.

Builds a `faster_qwen3_tts.FasterQwen3TTS` (CUDA-graph decoding) and picks a
configuration for the GPU it lands on. On a large modern GPU this is the stock
setup (bf16, everything on GPU); on small or older cards it enables:

* `GemvLinear`: on GPUs without tensor cores (e.g. GTX 16-series) cuBLAS picks
  pathologically slow fp16 kernels for `F.linear` (~25 GB/s). Pre-transposed
  weights hit fast GEMV kernels (~130 GB/s); small multi-row inputs are split
  into GEMVs and large ones are computed in fp32. Chosen by microbenchmark.
* CPU text embedding: the 311M-param table is only indexed at prefill, so
  offloading it saves ~0.6 GiB of VRAM at negligible cost.
* CPU speech decoder: used when it doesn't fit in VRAM. The decoder overflows
  to NaN in fp16, so it runs in bf16 on GPUs that support it, else fp32.

Weights always load on CPU first and only the planned parts move to the GPU, so
loading never exceeds the VRAM budget (on Windows that would silently spill into
slow shared memory; elsewhere it would OOM).
"""
import copy
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.predictor_graph import PredictorGraph
from faster_qwen3_tts.talker_graph import TalkerGraph
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LOCAL_MODEL = Path(__file__).resolve().parent / "model"
GiB = 2**30
RUNTIME_RESERVE = 0.25 * GiB  # CUDA graph pools, predictor cache, activations (measured ~0.1 GiB + KV cache)
DECODER_ACTIVATIONS = 1.0 * GiB  # decoder conv activations; grows with utterance length
# Under Windows WDDM, mem_get_info understates what can be allocated before the driver
# spills into shared system memory (measured: 3.22 GiB reported vs ~3.75 GiB usable on 4 GiB).
WDDM_SLACK = 0.4 * GiB if os.name == "nt" else 0


class GemvLinear(torch.nn.Linear):
    """nn.Linear storing W^T contiguously and routing around slow half-precision GEMMs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        wt = self.weight_t
        if x2.shape[0] == 1:
            y = x2 @ wt
        elif x2.shape[0] <= 4:
            y = torch.cat([row @ wt for row in x2.split(1)])
        else:
            y = (x2.float() @ wt.float()).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], -1)

    @classmethod
    def convert(cls, module: torch.nn.Module) -> None:
        for m in module.modules():
            if type(m) is torch.nn.Linear:
                wt = m.weight.detach().t().contiguous()
                del m.weight
                m.register_buffer("weight_t", wt)
                m.__class__ = cls


class OffloadedEmbedding(torch.nn.Module):
    """Embedding lookup on CPU, result shipped to `device`.

    The table is held outside the module tree so `.to()` never moves it.
    """

    def __init__(self, emb: torch.nn.Embedding, device: str):
        super().__init__()
        self._emb = [emb.cpu()]
        self.device = device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._emb[0](ids.cpu()).to(self.device, non_blocking=True)


@dataclass
class Plan:
    dtype: torch.dtype
    gemv: bool
    offload_embedding: bool
    decoder_on_gpu: bool

    @property
    def decoder_dtype(self) -> torch.dtype:
        """fp16 overflows to NaN; CPU half-precision convs are very slow."""
        return torch.bfloat16 if self.decoder_on_gpu and self.dtype == torch.bfloat16 else torch.float32

    def __str__(self) -> str:
        dec = f"{'GPU' if self.decoder_on_gpu else 'CPU'} {str(self.decoder_dtype)[6:]}"
        return (f"dtype={str(self.dtype)[6:]}, gemv_linear={self.gemv}, "
                f"text_embedding={'CPU' if self.offload_embedding else 'GPU'}, decoder={dec}")


def _nbytes(module: torch.nn.Module, dtype: Optional[torch.dtype] = None) -> int:
    tensors = list(module.parameters()) + list(module.buffers())
    if dtype is None:
        return sum(t.numel() * t.element_size() for t in tensors)
    size = torch.empty(0, dtype=dtype).element_size()
    return sum(t.numel() * (size if t.is_floating_point() else t.element_size()) for t in tensors)


def _gemv_is_faster(device: str, dtype: torch.dtype) -> bool:
    """Microbenchmark a representative MLP projection for 1- and 2-row inputs."""
    ref = torch.nn.Linear(2048, 6144, bias=False, device=device, dtype=dtype)
    alt = copy.deepcopy(ref)
    GemvLinear.convert(alt)
    xs = [torch.randn(m, 2048, device=device, dtype=dtype) for m in (1, 2)]

    def bench(layer: torch.nn.Module) -> float:
        for x in xs:
            layer(x)
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        for _ in range(50):
            for x in xs:
                layer(x)
        torch.cuda.synchronize(device)
        return time.perf_counter() - t

    with torch.inference_mode():
        return bench(alt) < 0.8 * bench(ref)


def _plan(base: Qwen3TTSModel, device: str, dtype: torch.dtype, max_seq_len: int,
          gemv: Optional[bool], offload_embedding: Optional[bool], decoder_on_gpu: Optional[bool]) -> Plan:
    """Fill unset options: the decoder gets VRAM before the text embedding,
    since CPU decoding costs far more time than a CPU embedding lookup."""
    if gemv is None:
        gemv = _gemv_is_faster(device, dtype)
    plan = Plan(dtype, gemv, bool(offload_embedding), bool(decoder_on_gpu))
    if offload_embedding is not None and decoder_on_gpu is not None:
        return plan

    tcfg = base.model.config.talker_config
    kv_cache = 2 * tcfg.num_hidden_layers * tcfg.num_key_value_heads * tcfg.head_dim * max_seq_len * dtype.itemsize
    emb = _nbytes(base.model.talker.model.text_embedding)
    core = _nbytes(base.model) - emb
    decoder_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    decoder = _nbytes(base.model.speech_tokenizer.model, decoder_dtype) + DECODER_ACTIVATIONS

    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(device)[0] + WDDM_SLACK
    spare = free - core - kv_cache - RUNTIME_RESERVE
    if spare < (0 if offload_embedding is not False else emb):
        warnings.warn(f"Qwen3-TTS needs ~{(core + kv_cache + RUNTIME_RESERVE) / GiB:.1f} GiB of free VRAM on {device}, "
                      f"~{free / GiB:.1f} GiB available; expect out-of-memory errors or (on Windows) slow "
                      "shared-memory spill. Close other GPU apps or lower max_seq_len.", stacklevel=3)
    if decoder_on_gpu is None:
        plan.decoder_on_gpu = spare - (emb if offload_embedding is False else 0) >= decoder
    if plan.decoder_on_gpu:
        spare -= decoder
    if offload_embedding is None:
        plan.offload_embedding = spare < emb
    return plan


def load(model_id: Optional[str] = None, device: str = "cuda", max_seq_len: int = 1024, *,
         dtype: Optional[torch.dtype] = None, gemv: Optional[bool] = None,
         offload_embedding: Optional[bool] = None, decoder_on_gpu: Optional[bool] = None) -> FasterQwen3TTS | Qwen3TTSModel:
    """Load the local model if available, or the default Hub model.

    On CUDA, pick a plan for the GPU and capture CUDA graphs. On CPU, use
    standard Qwen3-TTS generation in float32 unless dtype is overridden.

    `max_seq_len` bounds prompt + audio frames (12 frames/s); smaller is faster
    since attention spans the whole static cache. The keyword options override
    auto-detection; the chosen configuration is stored on `tts.plan`.
    """
    if model_id is None:
        model_id = str(LOCAL_MODEL) if (LOCAL_MODEL / "model.safetensors").is_file() else MODEL_ID
    if torch.device(device).type == "cpu":
        base = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu", dtype=dtype or torch.float32,
                                            attn_implementation="sdpa")
        base.plan = f"CPU {str(dtype or torch.float32)[6:]}, standard Qwen3-TTS generation"
        return base
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with NVIDIA GPU access or choose device='cpu'.")
    device = str(torch.device(device if ":" in device else f"{device}:{torch.cuda.current_device()}"))
    if dtype is None:
        # bf16 is the training dtype; Turing and older only emulate it.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    base = Qwen3TTSModel.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
    plan = _plan(base, device, dtype, max_seq_len, gemv, offload_embedding, decoder_on_gpu)

    tokenizer = base.model.speech_tokenizer
    tokenizer.model.to(device if plan.decoder_on_gpu else "cpu", plan.decoder_dtype)
    tokenizer.device = torch.device(device if plan.decoder_on_gpu else "cpu")
    base.sample_rate = tokenizer.get_output_sample_rate()

    talker = base.model.talker
    emb = talker.model.text_embedding
    if plan.offload_embedding:
        talker.model.text_embedding = None
    if plan.gemv:
        GemvLinear.convert(talker)
    base.model.to(device)
    base.device = torch.device(device)  # cached at construction; used to place input ids
    if plan.offload_embedding:
        talker.model.text_embedding = OffloadedEmbedding(emb, device)

    tcfg = base.model.config.talker_config
    predictor_graph = PredictorGraph(talker.code_predictor, talker.code_predictor.model.config, tcfg.hidden_size,
                                     device=device, dtype=dtype, do_sample=True, top_k=50, temperature=0.9)
    talker_graph = TalkerGraph(talker.model, tcfg, device=device, dtype=dtype, max_seq_len=max_seq_len)
    tts = FasterQwen3TTS(base, predictor_graph, talker_graph, device=device, dtype=dtype, max_seq_len=max_seq_len)
    tts.plan = plan
    tts.warmup()
    return tts
