"""Durable, model-independent NPC reference recordings for audio.cpp."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from reference_audio import ReferenceSelector


class SpeechEngine:
    """Called on the serialized NPC worker; model weights live in audio.cpp."""

    def __init__(self, backend, config, voices_dir, selector=None):
        self.backend, self.config = backend, config
        self.voices_dir = Path(voices_dir)
        self.selector = selector or ReferenceSelector(model_path=config.alignment_model,
            min_seconds=config.reference_min_seconds, max_seconds=config.reference_max_seconds)

    def voice_path(self, npc_id):
        return self.voices_dir / hashlib.sha256(npc_id.encode()).hexdigest()

    def record(self, npc_id):
        folder = self.voice_path(npc_id)
        path = folder / "metadata.json"
        return folder, json.loads(path.read_text()) if path.is_file() else None

    def status(self, npc_id):
        folder, metadata = self.record(npc_id)
        if metadata is None:
            return "missing"
        default = "ready" if (folder / "reference.wav").is_file() and metadata.get("ref_text") else "preparing"
        status = metadata.get("status", default)
        return default if status == "ready" else status

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

    def save_dialogue(self, body, model_id, description, seed, audio, rate):
        folder = self.voice_path(body.npc_id)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"version": 3, "status": "preparing", "npc_id": body.npc_id,
                    "description": description, "seed": seed, "dialogue_text": body.input,
                    "dialogue_language": body.language, "sample_rate": rate,
                    "model": model_id, "backend": "audio.cpp"}
        with tempfile.TemporaryDirectory(prefix=".creating-", dir=self.voices_dir) as temporary:
            path = Path(temporary)
            sf.write(path / "dialogue.wav", audio, rate, subtype="PCM_16")
            self.write_metadata(path, metadata)
            os.replace(path, folder)

    def prepare(self, npc_id, model_id=None):
        folder, metadata = self.record(npc_id)
        if metadata is None:
            raise ValueError("Unknown NPC")
        if self.status(npc_id) == "ready" and (folder / "reference.wav").is_file() and metadata.get("ref_text"):
            return
        saved_model = metadata.get("model")
        if saved_model not in self.config.models and saved_model not in self.config.aliases:
            saved_model = None
        model_id = model_id or saved_model or self.config.default_model
        _, profile = self.config.resolve(model_id)
        if not profile.clone_model:
            raise ValueError("This profile does not support NPC cloning")
        metadata.update(status="preparing")
        metadata.pop("error", None)
        self.write_metadata(folder, metadata)
        try:
            audio, rate = sf.read(folder / "dialogue.wav", dtype="float32")
            try:
                segment, transcript, source = self.selector.select(audio, rate, metadata["dialogue_text"],
                    metadata["dialogue_language"], may_be_truncated=metadata.get("may_be_truncated", False))
                language = metadata["dialogue_language"]
            except (ValueError, RuntimeError, ImportError):
                from server import SpeechRequest
                body = SpeechRequest(model=model_id, input=self.config.reference_extension_text,
                    language=self.config.reference_extension_language,
                    seed=metadata["seed"] % (profile.seed_max + 1), response_format="wav")
                # A complete short recording has a known transcript. A cropped long
                # recording does not; passing its full text would misalign cloning.
                reference_text = None
                if len(audio) / rate <= self.config.reference_max_seconds and not metadata.get("may_be_truncated", False):
                    reference_text = metadata["dialogue_text"]
                extended, rate = self.backend.generate(profile, body, mode="clone",
                    reference=(audio[:round(rate * self.config.reference_max_seconds)], rate),
                    reference_text=reference_text,
                    options=self.config.reference_options)
                segment, transcript, _ = self.selector.select(extended, rate, body.input, body.language)
                source, language = "extended_dialogue", body.language
            duration = len(segment) / rate
            if not self.config.reference_min_seconds <= duration <= self.config.reference_max_seconds:
                raise RuntimeError("Reference does not fit the configured duration range")
            if not np.isfinite(segment).all() or np.max(np.abs(segment)) < 0.001:
                raise RuntimeError("Reference is silent or invalid")
            sf.write(folder / ".reference.wav", segment, rate, subtype="PCM_16")
            os.replace(folder / ".reference.wav", folder / "reference.wav")
            metadata.update(version=3, status="ready", ref_text=transcript, ref_language=language,
                            sample_rate=rate, reference_seconds=duration, reference_source=source,
                            backend="audio.cpp", model=model_id)
            self.write_metadata(folder, metadata)
        except Exception:
            metadata.update(status="failed", error="Reference preparation failed; a later speech request will retry.")
            self.write_metadata(folder, metadata)
            raise

    def generate(self, body, model_id, profile):
        if not body.npc_id:
            voice = body.voice or profile.default_voice
            instruction = body.instructions or profile.voice_descriptions.get(voice)
            mode = "design" if instruction and profile.design_model else "speech"
            return self.backend.generate(profile, body, mode=mode, instruction=instruction)
        if not profile.clone_model:
            raise ValueError("This profile does not support NPC cloning")
        folder, metadata = self.record(body.npc_id)
        if metadata is None:
            if not profile.design_model or not body.instructions or not body.instructions.strip():
                raise ValueError("A new NPC needs a voice-design profile and instructions")
            seed = body.seed if body.seed is not None else int(hashlib.sha256(body.npc_id.encode()).hexdigest()[:15], 16) % (profile.seed_max + 1)
            audio, rate = self.backend.generate(profile, body, mode="design", instruction=body.instructions, seed=seed)
            if np.max(np.abs(audio)) < 0.001:
                raise RuntimeError("First dialogue is silent")
            self.save_dialogue(body, model_id, body.instructions, seed, audio, rate)
            return audio, rate
        self.prepare(body.npc_id, model_id)
        folder, metadata = self.record(body.npc_id)
        reference, rate = sf.read(folder / "reference.wav", dtype="float32")
        return self.backend.generate(profile, body, mode="clone", reference=(reference, rate),
                                     reference_text=metadata["ref_text"])
