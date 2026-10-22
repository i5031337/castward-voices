"""Public speech profiles mapped to independently managed audio.cpp models."""
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend_model: str
    display_name: str | None = None
    description: str | None = None
    design_model: str | None = None
    clone_model: str | None = None
    streaming_model: str | None = None
    streaming_sample_rate: int | None = Field(default=None, gt=0)
    streaming_channels: int = Field(default=1, gt=0)
    instruction_option: str | None = None
    reference_text_option: str = "reference_text"
    speaker_only_option: str | None = None
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

    @staticmethod
    def conditions_match(options, conditions):
        return all(options.get(key) in allowed for key, allowed in conditions.items())

    def accepts_instruction(self, options):
        return self.conditions_match(options, self.instruction_conditions)

    def validate_options(self, options):
        for name, conditions in self.option_dependencies.items():
            if name in options and not self.conditions_match(options, conditions):
                required = "; ".join(f"{key}={','.join(map(str, allowed))}" for key, allowed in conditions.items())
                raise ValueError(f"{name} requires {required}. Change the template or turn off the {name} override.")
        if self.instruction_option and options.get(self.instruction_option) and not self.accepts_instruction(options):
            raise ValueError("This template uses structured controls instead of a voice description. Clear the instruction override or choose a description-based template.")


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend_url: str = "http://127.0.0.1:7861"
    backend_timeout: float = Field(default=180, gt=0)
    backend_api_key_env: str | None = None
    output_sample_rate: int = Field(default=24000, gt=0)
    default_model: str
    aliases: dict[str, str] = Field(default_factory=dict)
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
        return self

    def resolve(self, name):
        key = self.aliases.get(name, name)
        if key not in self.models:
            raise ValueError(f"Unknown model: {name}")
        return key, self.models[key]


def load_config(path):
    path = Path(path).resolve()
    settings = ServerConfig.model_validate_json(path.read_text())
    if settings.alignment_model:
        alignment = Path(settings.alignment_model)
        settings.alignment_model = str(alignment if alignment.is_absolute() else path.parent / alignment)
    return settings
