"""Prepare temporary native/facade configs for CPU Kokoro and CUDA character speech."""
import argparse
import copy
import json
from pathlib import Path
import socket

from backend_config import load_config


def prepare_configs(native_path, speech_path, output_dir, *, gpu_device=0, gpu_port=7864, cpu_port=7863):
    if gpu_device < 0:
        raise ValueError("GPU device must be nonnegative")
    if gpu_port == cpu_port or any(not 1 <= port <= 65535 for port in (gpu_port, cpu_port)):
        raise ValueError("Native CPU/GPU ports must be different and between 1 and 65535")
    native_path = Path(native_path).resolve()
    native = json.loads(native_path.read_text())
    speech = load_config(speech_path).model_dump()
    if speech["backend_routes"]:
        raise ValueError("--kokoro-cpu expects one source backend; remove custom backend_routes or launch those services manually")
    models = copy.deepcopy(native["models"])
    # Native model paths are relative to the source config, not the temporary directory.
    for model in models:
        for key in ("path", "model_spec_override"):
            if model.get(key) and model[key] != "embedded":
                path = Path(model[key])
                model[key] = str(path if path.is_absolute() else native_path.parent / path)
        for preset in model.get("voice_presets", {}).values():
            if isinstance(preset.get("voice_ref"), str):
                path = Path(preset["voice_ref"])
                preset["voice_ref"] = str(path if path.is_absolute() else native_path.parent / path)
    cpu_models = [model for model in models if model.get("family") == "kokoro_tts"]
    gpu_models = [model for model in models if model.get("family") != "kokoro_tts"]
    if not cpu_models or not gpu_models:
        raise ValueError("--kokoro-cpu needs Kokoro and at least one GPU model in the native config")
    cpu_ids = {model["id"] for model in cpu_models}
    gpu_url, cpu_url = f"http://127.0.0.1:{gpu_port}", f"http://127.0.0.1:{cpu_port}"
    gpu = copy.deepcopy(native)
    gpu.update(host="127.0.0.1", port=gpu_port, backend="cuda", device=gpu_device,
               lazy_load=True, max_loaded_models=1, models=gpu_models)
    for model in gpu_models:
        model["device"] = gpu_device
        model["lazy"] = True
    cpu = copy.deepcopy(native)
    cpu.update(host="127.0.0.1", port=cpu_port, backend="cpu", lazy_load=True,
               max_loaded_models=len(cpu_models), models=cpu_models)
    cpu.pop("device", None)
    for model in cpu_models:
        model.pop("device", None)
        model["lazy"] = True
    speech["backend_url"] = gpu_url
    speech["backend_routes"] = {model: cpu_url for model in sorted(cpu_ids)}
    for profile in speech["models"].values():
        if profile.get("streaming_model") in cpu_ids and "text_chunk_size" not in profile["options"]:
            profile["streaming_options"].setdefault("text_chunk_size", 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, settings in (("gpu.json", gpu), ("cpu.json", cpu), ("backend.json", speech)):
        (output_dir / filename).write_text(json.dumps(settings, indent=2) + "\n")


def check_ports(*ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise ValueError(f"CPU/GPU native port is occupied: {exc}. Set AUDIOCPP_GPU_PORT and AUDIOCPP_CPU_PORT to unused ports.") from exc
    finally:
        for sock in sockets:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_config", type=Path)
    parser.add_argument("speech_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-port", type=int, default=7864)
    parser.add_argument("--cpu-port", type=int, default=7863)
    args = parser.parse_args()
    try:
        prepare_configs(args.native_config, args.speech_config, args.output_dir,
                        gpu_device=args.gpu_device, gpu_port=args.gpu_port, cpu_port=args.cpu_port)
        check_ports(args.gpu_port, args.cpu_port)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot launch CPU Kokoro: {exc}\n")


if __name__ == "__main__":
    main()
