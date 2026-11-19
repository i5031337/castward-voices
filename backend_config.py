"""Public speech profiles mapped to independently managed audio.cpp models."""
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend_model: str
    display_name: str | None = None
    description: str | None = None
    design_name: str | None = None
    clone_name: str | None = None
    listed: bool = True
    preset_enabled: bool = True
    preset_source: bool = False
    preset_default_voice: str | None = None
    mode_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    generated: bool = False
    design_model: str | None = None
    clone_model: str | None = None
    streaming_model: str | None = None
    streaming_sample_rate: int | None = Field(default=None, gt=0)
    streaming_channels: int = Field(default=1, gt=0)
    instruction_option: str | None = None
    reference_text_option: str | None = "reference_text"
    speaker_only_option: str | None = None
    max_tokens_option: str | None = "max_tokens"
    seed_max: int = Field(default=2**64-1, ge=0, le=2**64-1)
    language_option: str | None = "language"
    language_map: dict[str, str | None] = Field(default_factory=dict)
    default_voice: str | None = None
    voice_aliases: dict[str, str] = Field(default_factory=dict)
    voice_descriptions: dict[str, str] = Field(default_factory=dict)
    voice_option: str | None = None
    preset_options: dict[str, Any] = Field(default_factory=dict)
    option_dependencies: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    instruction_conditions: dict[str, list[Any]] = Field(default_factory=dict)
    mode_conditions: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    design_options: dict[str, Any] = Field(default_factory=dict)
    clone_options: dict[str, Any] = Field(default_factory=dict)
    streaming_options: dict[str, Any] = Field(default_factory=dict)
    context: bool = False
    context_max_chars: int = Field(default=240, gt=0)
    native_speed: bool = False
    max_input_chars: int = Field(default=16384, gt=0)

    @model_validator(mode="after")
    def valid_stream(self):
        if self.streaming_model and not self.streaming_sample_rate:
            raise ValueError("streaming_sample_rate is required for raw native streaming")
        return self

    def options_for(self, mode, overrides=None):
        defaults = {"speech": self.preset_options, "design": self.design_options,
                    "clone": self.clone_options, "stream": self.streaming_options}.get(mode, {})
        return {**self.options, **defaults, **(overrides or {})}

    def for_mode(self, mode):
        source = self.mode_profiles.get(mode)
        return ModelProfile.model_validate(source) if source else self

    @staticmethod
    def conditions_match(options, conditions):
        return all(options.get(key) in allowed for key, allowed in conditions.items())

    def accepts_instruction(self, options):
        return self.conditions_match(options, self.instruction_conditions)

    def validate_options(self, options, mode=None):
        conditions = self.mode_conditions.get(mode, {})
        if not self.conditions_match(options, conditions):
            required = "; ".join(f"{key}={','.join(map(str, allowed))}" for key, allowed in conditions.items())
            raise ValueError(f"{mode} mode requires {required} for this profile")
        for name, conditions in self.option_dependencies.items():
            if name in options and not self.conditions_match(options, conditions):
                required = "; ".join(f"{key}={','.join(map(str, allowed))}" for key, allowed in conditions.items())
                raise ValueError(f"{name} requires {required}. Change the template or turn off the {name} override.")
        if self.instruction_option and options.get(self.instruction_option) and not self.accepts_instruction(options):
            raise ValueError("This template uses structured controls instead of a voice description. Clear the instruction override or choose a description-based template.")


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend_url: str = "http://127.0.0.1:7861"
    backend_routes: dict[str, str] = Field(default_factory=dict)
    backend_timeout: float = Field(default=180, gt=0)
    backend_api_key_env: str | None = None
    output_sample_rate: int = Field(default=24000, gt=0)
    output_level_dbfs: float | None = Field(default=-20.0, ge=-35, le=-10)
    default_model: str
    aliases: dict[str, str] = Field(default_factory=dict)
    auto_pairs: bool = False
    models: dict[str, ModelProfile]
    reference_min_seconds: float = Field(default=3, gt=0)
    reference_max_seconds: float = Field(default=10, gt=0)
    reference_extension_text: str = "Welcome, traveler. The road ahead is long, but there is always a place for you by the fire."
    reference_extension_language: str = "English"
    reference_options: dict[str, Any] = Field(default_factory=dict)
    alignment_model: str | None = "models/whisper/tiny.pt"

    @model_validator(mode="after")
    def valid_models(self):
        if self.default_model not in self.models:
            raise ValueError("default_model must name a configured profile")
        if any(target not in self.models for target in self.aliases.values()):
            raise ValueError("Every alias must target a configured profile")
        if set(self.aliases) & set(self.models):
            raise ValueError("Aliases must not shadow profile IDs")
        if self.reference_min_seconds > self.reference_max_seconds:
            raise ValueError("Invalid reference duration range")
        if self.auto_pairs:
            self.build_pair_profiles()
        return self

    def build_pair_profiles(self):
        sources = [profile for profile in self.models.values() if not profile.generated]
        designers, cloners = {}, {}
        for profile in sources:
            if profile.design_model:
                designers.setdefault(profile.design_model, profile)
            if profile.clone_model:
                cloners.setdefault(profile.clone_model, profile)
        for profile in sources:
            if profile.design_model and not profile.preset_enabled:
                profile.listed = False
            if profile.preset_source:
                name = f"{profile.backend_model}-presets"
                if name not in self.models:
                    preset = profile.model_copy(deep=True)
                    preset.design_model = None
                    preset.clone_model = None
                    preset.preset_enabled = True
                    preset.preset_source = False
                    preset.default_voice = profile.preset_default_voice
                    preset.voice_descriptions = {}
                    preset.listed = True
                    preset.generated = True
                    preset.display_name = f"{profile.design_name or profile.backend_model} · preset voices"
                    preset.description = f"Preset synthesis with {profile.design_name or profile.backend_model}."
                    self.models[name] = preset
        for design_id, designer in designers.items():
            for clone_id, cloner in cloners.items():
                name = f"design-{design_id}--clone-{clone_id}"
                if name in self.models:
                    continue
                pair = designer.model_copy(deep=True)
                pair.design_model, pair.clone_model = design_id, clone_id
                design_name = designer.design_name or design_id
                clone_name = cloner.clone_name or clone_id
                pair.display_name = f"Design: {design_name} · Clone: {clone_name}"
                pair.description = f"Design with {design_name}; clone the resulting voice with {clone_name}."
                pair.preset_enabled = False
                pair.listed = True
                pair.generated = True
                pair.mode_profiles = {
                    "design": designer.model_dump(exclude={"mode_profiles"}),
                    "clone": cloner.model_dump(exclude={"mode_profiles"}),
                }
                pair.seed_max = min(designer.seed_max, cloner.seed_max)
                pair.max_input_chars = min(designer.max_input_chars, cloner.max_input_chars)
                self.models[name] = pair

    def public_models(self):
        return {name: profile for name, profile in self.models.items() if profile.listed}

    def public_default_model(self):
        if self.models[self.default_model].listed:
            return self.default_model
        source = self.models[self.default_model]
        pair = f"design-{source.design_model}--clone-{source.clone_model}"
        return pair if pair in self.models else next(iter(self.public_models()))

    def resolve(self, name):
        key = self.aliases.get(name, name)
        if key not in self.models:
            raise ValueError(f"Unknown model: {name}")
        return key, self.models[key]


def load_config(path):
    path = Path(path).resolve()
    settings = ServerConfig.model_validate_json(path.read_bytes())
    if settings.alignment_model:
        alignment = Path(settings.alignment_model)
        settings.alignment_model = str(alignment if alignment.is_absolute() else path.parent / alignment)
    return settings
