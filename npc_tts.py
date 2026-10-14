"""Persistent NPC references with optional residency for both Qwen models."""
import gc
import hashlib
import json
import logging
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
from narrator_tts import MODEL_LOAD_LOCK
from reference_audio import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS, ReferenceSelector

CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LOCAL_CLONE_MODEL = Path(__file__).resolve().parent / "models" / "0.6B-Base"
REFERENCE_TEXT = ("Welcome, traveler. The road ahead is long, but there is always "
                  "a place for you by the fire.")
log = logging.getLogger("qwen3-tts.npc")


class SpeechEngine:
    """Called only on the server's dedicated generation thread."""

    def __init__(self, *, loader, device, design_model, clone_model, max_seq_len, voices_dir,
                 selector=None, clone_device=None, keep_models_loaded=False):
        self.loader = loader
        self.device = device
        self.clone_device = clone_device or device
        self.active_device = None
        self.design_model = design_model
        self.clone_model = clone_model
        self.max_seq_len = max_seq_len
        self.voices_dir = Path(voices_dir)
        self.tts = None
        self.kind = None
        self.keep_models_loaded = keep_models_loaded
        self.loaded_models = {}
        self.prompts = OrderedDict()
        self.selector = selector or ReferenceSelector()
        self.get("clone" if clone_model else "design")
        if keep_models_loaded and clone_model:
            # Capture both models' graphs before accepting requests, then keep
            # Base selected for normal NPC dialogue. Loading the second model
            # accounts for the first model's live allocations in its VRAM plan.
            self.get("design")
            self.get("clone")

    @property
    def plan(self):
        return self.tts.plan

    def get(self, kind):
        if kind not in {"clone", "design"}:
            raise ValueError("Unknown Qwen model kind")
        if kind == self.kind:
            return self.tts
        if kind == "clone" and not self.clone_model:
            raise ValueError("NPC cloning is disabled on this server")
        target_device = self.clone_device if kind == "clone" else self.device
        if kind in self.loaded_models:
            self.tts = self.loaded_models[kind]
            self.active_device = target_device
            self.kind = kind
            return self.tts
        # Default mode releases graphs/weights. Resident mode keeps the previous
        # model alive in loaded_models while constructing the other pipeline.
        with MODEL_LOAD_LOCK:
            self.tts = None
            self.kind = None
            gc.collect()
            import torch
            if self.device.startswith("cuda"):
                torch.cuda.set_device(self.device)
                torch.cuda.empty_cache()
            model = self.clone_model if kind == "clone" else self.design_model
            self.tts = self.loader(model_id=model, device=target_device, max_seq_len=self.max_seq_len)
            if self.keep_models_loaded:
                self.loaded_models[kind] = self.tts
        self.active_device = target_device
        self.kind = kind
        log.info("Loaded %s model: %s", kind, self.tts.plan)
        return self.tts

    def check_budget(self, tts, *, text, language, max_new_tokens, instruct=None, prompt=None, ref_text=""):
        if not self.active_device.startswith("cuda"):
            return
        if prompt is None:
            prepared = tts._prepare_generation_custom(text=text, language=language, speaker=None,
                                                     instruct=instruct, non_streaming_mode=True)
        else:
            prepared = tts._prepare_generation(text=text, language=language, ref_text=ref_text,
                                               voice_clone_prompt=prompt, non_streaming_mode=True)
        length = prepared[3].shape[1]
        if length + max_new_tokens >= self.max_seq_len:
            raise ValueError("Prompt plus max_new_tokens exceeds --max-seq-len. "
                             "Lower max_new_tokens or start the server with a larger sequence budget.")

    def voice_path(self, npc_id):
        return self.voices_dir / hashlib.sha256(npc_id.encode("utf-8")).hexdigest()

    def record(self, npc_id):
        folder = self.voice_path(npc_id)
        if (folder / "metadata.json").is_file():
            return folder, json.loads((folder / "metadata.json").read_text())
        return folder, None

    def status(self, npc_id):
        folder, metadata = self.record(npc_id)
        if metadata is None:
            return "missing"
        return metadata.get("status", "ready" if (folder / "prompt.npz").is_file() else "preparing")

    def pending_ids(self):
        for path in self.voices_dir.glob("*/metadata.json"):
            metadata = json.loads(path.read_text())
            if metadata.get("status") == "preparing":
                yield metadata["npc_id"]

    @staticmethod
    def write_metadata(folder, metadata):
        temporary = folder / ".metadata.json"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, folder / "metadata.json")

    def save_dialogue(self, body, description, seed, wav, sr):
        """Publish the first dialogue before expensive clone preparation begins."""
        folder = self.voice_path(body.npc_id)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"version": 2, "status": "preparing", "npc_id": body.npc_id,
                    "description": description, "seed": seed, "dialogue_text": body.input,
                    "dialogue_language": body.language, "sample_rate": sr,
                    "may_be_truncated": len(wav) / sr >= max(0, body.max_new_tokens / 12 - 0.5),
                    "clone_model": CLONE_MODEL_ID}
        with tempfile.TemporaryDirectory(prefix=".creating-", dir=self.voices_dir) as staging:
            path = Path(staging)
            sf.write(path / "dialogue.wav", wav, sr, subtype="PCM_16")
            self.write_metadata(path, metadata)
            os.replace(path, folder)

    @staticmethod
    def base(tts):
        return tts.model if hasattr(tts, "predictor_graph") else tts

    def extend_reference(self, tts, wav, sr, seed):
        """Extend an inadequate first line using its speaker embedding, not a new voice design."""
        import torch
        base = self.base(tts)
        items = base.create_voice_clone_prompt(ref_audio=(wav[:round(sr * MAX_REFERENCE_SECONDS)], sr),
                                               ref_text="", x_vector_only_mode=True)
        prompt = base._prompt_items_to_voice_clone_prompt(items)
        self.check_budget(tts, text=REFERENCE_TEXT, language="English", max_new_tokens=192, prompt=prompt)
        torch.manual_seed(seed)
        if not self.active_device.startswith("cuda"):
            prompt = items
        wavs, out_sr = tts.generate_voice_clone(text=REFERENCE_TEXT, language="English",
            voice_clone_prompt=prompt, ref_text="", non_streaming_mode=True, temperature=0.9, max_new_tokens=192)
        extended = np.asarray(wavs[0])
        segment, text, _ = self.selector.select(extended, out_sr, REFERENCE_TEXT, "English",
                                               may_be_truncated=len(extended) / out_sr >= 192 / 12 - 0.5)
        return segment, out_sr, text

    def prepare(self, npc_id):
        """Idempotently finish a pending record; run on the generation worker after response delivery."""
        folder, metadata = self.record(npc_id)
        if metadata is None:
            raise ValueError("Unknown NPC")
        if self.status(npc_id) == "ready":
            return
        import torch
        metadata["status"] = "preparing"
        metadata.pop("error", None)
        self.write_metadata(folder, metadata)
        try:
            with torch.inference_mode():
                self._prepare(folder, metadata)
        except Exception:
            metadata["status"] = "failed"
            metadata["error"] = "Reference preparation failed; a later speech request will retry."
            self.write_metadata(folder, metadata)
            raise

    def _prepare(self, folder, metadata):
        wav, sr = sf.read(folder / "dialogue.wav", dtype="float32")
        try:
            segment, transcript, source = self.selector.select(wav, sr, metadata["dialogue_text"],
                metadata["dialogue_language"], may_be_truncated=metadata.get("may_be_truncated", False))
            reference_language = metadata["dialogue_language"]
        except (ValueError, RuntimeError) as exc:
            log.info("Using an extended reference for NPC %s: %s", metadata["npc_id"], exc)
            segment = None
        # Alignment runs on CPU before switching model weights, so no extra GPU model is loaded.
        clone = self.get("clone")
        if segment is None:
            segment, sr, transcript = self.extend_reference(clone, wav, sr, metadata["seed"])
            source, reference_language = "extended_dialogue", "English"
        if not np.isfinite(segment).all() or not MIN_REFERENCE_SECONDS <= len(segment) / sr <= MAX_REFERENCE_SECONDS:
            raise RuntimeError("Reference did not fit the duration budget")
        if np.abs(segment).max() < 0.001:
            raise RuntimeError("Reference is silent")
        base = self.base(clone)
        items = base.create_voice_clone_prompt(ref_audio=(segment, sr), ref_text=transcript,
                                               x_vector_only_mode=False)
        item = items[0]
        # Publish complete features before marking ready. Preserve the first dialogue
        # even when alignment or extraction fails, so retries keep the original voice.
        sf.write(folder / ".reference.wav", segment, sr, subtype="PCM_16")
        with (folder / ".prompt.npz").open("wb") as output:
            np.savez_compressed(output, ref_code=item.ref_code.cpu().numpy(),
                                ref_spk_embedding=item.ref_spk_embedding.float().cpu().numpy())
        os.replace(folder / ".reference.wav", folder / "reference.wav")
        os.replace(folder / ".prompt.npz", folder / "prompt.npz")
        metadata.update(status="ready", ref_text=transcript, ref_language=reference_language,
                        sample_rate=sr, reference_seconds=len(segment) / sr, reference_source=source)
        self.write_metadata(folder, metadata)
        log.info("Prepared %.2fs reference for NPC %s", len(segment) / sr, metadata["npc_id"])

    def prompt(self, npc_id, folder, metadata):
        if npc_id in self.prompts:
            self.prompts.move_to_end(npc_id)
            return self.prompts[npc_id]
        if metadata.get("version") not in {1, 2} or metadata.get("clone_model") != CLONE_MODEL_ID:
            raise ValueError("Unsupported saved voice record")
        import torch
        with np.load(folder / "prompt.npz", allow_pickle=False) as data:
            prompt = {"ref_code": [torch.from_numpy(data["ref_code"].copy()).long()],
                      "ref_spk_embedding": [torch.from_numpy(data["ref_spk_embedding"].copy())],
                      "x_vector_only_mode": [False], "icl_mode": [True]}
        self.prompts[npc_id] = prompt
        if len(self.prompts) > 128:
            self.prompts.popitem(last=False)
        return prompt

    def generate(self, body, description):
        import torch
        with torch.inference_mode():
            if body.npc_id:
                folder, metadata = self.record(body.npc_id)
                if metadata is None:
                    if not description or not description.strip():
                        raise ValueError("A new NPC needs instructions describing its voice")
                    seed = body.seed if body.seed is not None else int(hashlib.sha256(body.npc_id.encode()).hexdigest()[:15], 16)
                    tts = self.get("design")
                    self.check_budget(tts, text=body.input, language=body.language,
                                      max_new_tokens=body.max_new_tokens, instruct=description)
                    torch.manual_seed(seed)
                    wavs, sr = tts.generate_voice_design(text=body.input, instruct=description, language=body.language,
                                                        temperature=body.temperature, max_new_tokens=body.max_new_tokens)
                    wav = np.asarray(wavs[0])
                    if not wav.size or not np.isfinite(wav).all() or np.abs(wav).max() < 0.001:
                        raise RuntimeError("First dialogue is empty, invalid, or silent")
                    self.save_dialogue(body, description, seed, wav, sr)
                    return wavs, sr
                # A follow-up arriving before the post-response task queues must still
                # finish the same record, never generate another character voice.
                self.prepare(body.npc_id)
                folder, metadata = self.record(body.npc_id)
                tts = self.get("clone")
                prompt = self.prompt(body.npc_id, folder, metadata)
                self.check_budget(tts, text=body.input, language=body.language, max_new_tokens=body.max_new_tokens,
                                  prompt=prompt, ref_text=metadata["ref_text"])
                if body.seed is not None:
                    torch.manual_seed(body.seed)
                if not self.active_device.startswith("cuda"):
                    # The upstream CPU wrapper reads the reference transcript from
                    # prompt items, not from ref_text when a prompt dict is supplied.
                    from qwen_tts import VoiceClonePromptItem
                    prompt = [VoiceClonePromptItem(ref_code=prompt["ref_code"][0],
                        ref_spk_embedding=prompt["ref_spk_embedding"][0],
                        x_vector_only_mode=False, icl_mode=True, ref_text=metadata["ref_text"])]
                return tts.generate_voice_clone(text=body.input, language=body.language,
                    voice_clone_prompt=prompt, ref_text=metadata["ref_text"], non_streaming_mode=True,
                    temperature=body.temperature, max_new_tokens=body.max_new_tokens)
            tts = self.get("design")
            self.check_budget(tts, text=body.input, language=body.language, max_new_tokens=body.max_new_tokens,
                              instruct=description)
            if body.seed is not None:
                torch.manual_seed(body.seed)
            return tts.generate_voice_design(text=body.input, instruct=description, language=body.language,
                                            temperature=body.temperature, max_new_tokens=body.max_new_tokens)
