"""Install a shared native runtime and selected models from model_catalog.json."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
INSTALL = ROOT / ".cache/install"


def read_json(path):
    return json.loads(path.read_text())


def select_models(catalog, previous, models=None, add=None):
    selected = (list(catalog["models"]) if models == "all" else models.split(",")) if models else list(
        previous.get("models", catalog["default_models"]))
    if add:
        selected.extend(add.split(","))
    selected = list(dict.fromkeys(selected))
    unknown = set(selected) - catalog["models"].keys()
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}. Use --list-models.")
    if not selected:
        raise ValueError("Select at least one model")
    return selected


def make_configs(catalog, selected, *, cpu=False, device=0):
    native = copy.deepcopy(read_json(ROOT / "audiocpp-server.local.json"))
    speech = copy.deepcopy(read_json(ROOT / "backend.json"))
    weights, profiles = {}, set()
    for name in selected:
        model = catalog["models"][name]
        profiles.update(model["profiles"])
        weights.update({key: str(ROOT / model["local_dir"] / path)
                        for key, path in model["weights"].items()})
    native["models"] = [m for m in native["models"] if m["id"] in weights]
    for model in native["models"]:
        model["path"] = weights[model["id"]]
        model["lazy"] = True
        if cpu:
            model.pop("device", None)
        else:
            model["device"] = device
    native.update(host="127.0.0.1", backend="cpu" if cpu else "cuda", lazy_load=True,
                  max_loaded_models=1)
    native.pop("device", None)
    speech["backend_url"] = f"http://127.0.0.1:{native['port']}"
    speech["backend_routes"] = {}
    speech["models"] = {key: value for key, value in speech["models"].items() if key in profiles}
    speech["aliases"] = {key: value for key, value in speech.get("aliases", {}).items() if value in profiles}
    if speech["default_model"] not in profiles:
        speech["default_model"] = next(iter(speech["models"]))
    alignment = speech.get("alignment_model")
    speech["alignment_model"] = str(ROOT / alignment) if alignment and (ROOT / alignment).is_file() else None
    if set(weights) != {m["id"] for m in native["models"]} or profiles != speech["models"].keys():
        raise ValueError("Catalog references missing native models or public profiles")
    return native, speech


def build_commands(source, build, catalog, *, cpu, jobs):
    configure = ["cmake", "-S", str(source), "-B", str(build), "-DAUDIOCPP_MODEL_SET=custom",
                 "-DAUDIOCPP_MODELS=" + ",".join(dict.fromkeys(m["family"] for m in catalog["models"].values())),
                 "-DENGINE_ENABLE_CUDA=" + ("OFF" if cpu else "ON")]
    if not (build / "CMakeCache.txt").exists():
        configure.append("-DCMAKE_BUILD_TYPE=Release")
        if not cpu:
            configure.append("-DCMAKE_CUDA_ARCHITECTURES=" + os.environ.get("CUDA_ARCHITECTURES", "native"))
    elif not cpu and os.environ.get("CUDA_ARCHITECTURES"):
        configure.append("-DCMAKE_CUDA_ARCHITECTURES=" + os.environ["CUDA_ARCHITECTURES"])
    return [configure, ["cmake", "--build", str(build), "--target", "audiocpp_server", "--parallel", str(jobs)]]


def main():
    catalog = read_json(ROOT / "model_catalog.json")
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--models", help="Comma-separated models to enable, or all")
    selection.add_argument("--add", help="Add comma-separated models to the current installation")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--cpu", action="store_true", default=None, help="Build a CPU-only shared runtime")
    parser.add_argument("--gpu", action="store_true", help="Switch an existing CPU install to CUDA")
    parser.add_argument("--device", type=int, default=None, help="CUDA device (default: 0)")
    parser.add_argument("--plan", action="store_true", help="Show commands without installing")
    args = parser.parse_args()
    if args.list_models:
        for name, model in catalog["models"].items():
            print(f"{name:10} {', '.join(model['profiles'])}")
        return
    previous = read_json(INSTALL / "settings.json") if (INSTALL / "settings.json").exists() else {}
    try:
        selected = select_models(catalog, previous, args.models, args.add)
        if args.cpu and args.gpu:
            raise ValueError("Choose --cpu or --gpu")
        cpu = False if args.gpu else args.cpu or previous.get("cpu", False)
        device = args.device if args.device is not None else previous.get("device", 0)
        jobs = int(os.environ.get("BUILD_JOBS", "2"))
        if jobs < 1 or device < 0:
            raise ValueError("BUILD_JOBS must be positive and device nonnegative")
        # Validate catalog/config relationships before downloads or compilation.
        native, speech = make_configs(catalog, selected, cpu=cpu, device=device)
        existing_source = ROOT.parent / "Ming/audio.cpp"
        source = Path(os.environ.get("AUDIOCPP_SOURCE", previous.get("source", str(
            existing_source if existing_source.is_dir() else ROOT / ".cache/audio.cpp")))).resolve()
        same_backend = cpu == previous.get("cpu", False)
        default_build = previous.get("build") if same_backend else None
        build = Path(os.environ.get("AUDIOCPP_BUILD_DIR", default_build or str(
            ROOT / ".cache/audio.cpp-cpu-build" if cpu else source / "build/ming-cuda"))).resolve()
        commands = []
        if not source.exists():
            commands += [["git", "clone", catalog["source"]["url"], str(source)],
                         ["git", "-C", str(source), "checkout", catalog["source"]["revision"]],
                         ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"]]
        commands += [["uv", "venv", "--python", "3.11", str(ROOT / ".venv")]] if not (ROOT / ".venv").is_dir() else []
        commands += [["uv", "pip", "install", "--python", str(ROOT / ".venv/bin/python"), "-r", str(ROOT / "requirements.lock.txt")]]
        commands += build_commands(source, build, catalog, cpu=cpu, jobs=jobs)
        hf = ["hf"] if shutil.which("hf") else ["uv", "tool", "run", "--from", "huggingface_hub", "hf"]
        for name in selected:
            model = catalog["models"][name]
            patterns = model["include"]
            files = patterns if all(not any(c in path for c in "*?[") for path in patterns) else []
            command = hf + ["download", model["repo"], *files, "--revision", model["revision"],
                            "--local-dir", str(ROOT / model["local_dir"])]
            if not files:
                for pattern in patterns:
                    command += ["--include", pattern]
            if model.get("exclude"):
                command += ["--exclude", *model["exclude"]]
            commands.append(command)
        print("Models: " + ", ".join(selected), flush=True)
        if not args.plan:
            source.parent.mkdir(parents=True, exist_ok=True)
        for command in commands:
            print("+ " + shlex.join(command), flush=True)
            if not args.plan:
                subprocess.run(command, cwd=ROOT, check=True)
        if args.plan:
            return
        INSTALL.mkdir(parents=True, exist_ok=True)
        for name, data in (("native.json", native), ("backend.json", speech), ("settings.json", {
                "models": selected, "cpu": cpu, "device": device, "source": str(source), "build": str(build)})):
            temporary = INSTALL / (name + ".tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            temporary.replace(INSTALL / name)
        (INSTALL / "server-path").write_text(str(build / "bin/audiocpp_server") + "\n")
        print("Setup complete. Stop the running server, then launch: ./run-local.sh")
        if "pocket" in selected:
            print("PocketTTS attribution: THIRD_PARTY_NOTICES.md")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Setup failed: {exc}\nRerun the same command to resume completed downloads/build work.\n")


if __name__ == "__main__":
    main()
