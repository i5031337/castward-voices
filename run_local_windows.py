"""Start the native audio.cpp service and Castward API on Windows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import httpx

from launch_config import check_ports, prepare_configs


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
WINDOWS_PROCESS_FLAGS = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def installed_paths():
    install = CACHE / "install"
    if (install / "settings.json").is_file():
        return (Path((install / "server-path").read_text().strip()),
                install / "native.json", install / "backend.json")
    return (ROOT.parent / "Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server.exe",
            ROOT / "audiocpp-server.local.json", ROOT / "backend.json")


def healthy(url, expected=None):
    try:
        response = httpx.get(url.rstrip("/") + "/health", timeout=1)
        response.raise_for_status()
        health = response.json()
        return (isinstance(health, dict) and health.get("status") == "ok"
                and (expected is None or health.get("backend") == expected))
    except (httpx.HTTPError, ValueError):
        return False


def wait_native(url, process, logfile, expected=None):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"audio.cpp exited; see {logfile}")
        if healthy(url, expected):
            return
        time.sleep(0.25)
    raise RuntimeError(f"audio.cpp did not become ready; see {logfile}")


def start_process(command, logfile, env):
    log = (ROOT / logfile).open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   creationflags=WINDOWS_PROCESS_FLAGS)
    except BaseException:
        log.close()
        raise
    return process, log


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(argv=None):
    parser = argparse.ArgumentParser(description="Castward Voices local speech server")
    parser.add_argument("--kokoro-cpu", action="store_true",
                        help="CPU Kokoro/PocketTTS plus CUDA character speech")
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="GPU for CPU narration mode (default: 0)")
    parser.add_argument("--config", type=Path, help="Castward API config")
    args, facade_args = parser.parse_known_args(argv)
    native_binary, native_config, speech_config = installed_paths()
    native_binary = Path(os.environ.get("AUDIOCPP_SERVER", native_binary))
    native_config = Path(os.environ.get("AUDIOCPP_CONFIG", native_config))
    speech_config = args.config or Path(os.environ.get("SPEECH_CONFIG", speech_config))
    if not native_config.is_file() or not speech_config.is_file():
        raise RuntimeError("Native or API config is missing; run setup.bat first")
    env = os.environ.copy()
    if env.get("AUDIOCPP_CUDA_LIB"):
        env["PATH"] = env["AUDIOCPP_CUDA_LIB"] + os.pathsep + env.get("PATH", "")
    env.setdefault("XDG_CACHE_HOME", str(CACHE))
    children, logs = [], []
    temporary = None
    try:
        if args.kokoro_cpu:
            if not native_binary.is_file():
                raise RuntimeError(f"audio.cpp is missing: {native_binary}")
            gpu_port = int(env.get("AUDIOCPP_GPU_PORT", "7864"))
            cpu_port = int(env.get("AUDIOCPP_CPU_PORT", "7863"))
            CACHE.mkdir(exist_ok=True)
            temporary = tempfile.TemporaryDirectory(prefix="launch.", dir=CACHE)
            runtime = Path(temporary.name)
            prepare_configs(native_config, speech_config, runtime, gpu_device=args.gpu_device,
                            gpu_port=gpu_port, cpu_port=cpu_port)
            check_ports(gpu_port, cpu_port)
            gpu, log = start_process([str(native_binary), "--config", str(runtime / "gpu.json"),
                                      "--no-ui"], "audiocpp-backend-cpu-mode-gpu.log", env)
            children.append(gpu)
            logs.append(log)
            cpu, log = start_process([str(native_binary), "--config", str(runtime / "cpu.json"),
                                      "--no-ui"], "audiocpp-backend-cpu.log", env)
            children.append(cpu)
            logs.append(log)
            wait_native(f"http://127.0.0.1:{gpu_port}", gpu,
                        "audiocpp-backend-cpu-mode-gpu.log", "cuda")
            wait_native(f"http://127.0.0.1:{cpu_port}", cpu,
                        "audiocpp-backend-cpu.log", "cpu")
            speech_config = runtime / "backend.json"
            print(f"Kokoro/PocketTTS: CPU; character speech: CUDA device {args.gpu_device}",
                  flush=True)
        else:
            backend_url = json.loads(speech_config.read_text())["backend_url"]
            if not healthy(backend_url):
                if not native_binary.is_file():
                    raise RuntimeError(f"audio.cpp is missing: {native_binary}")
                native, log = start_process([str(native_binary), "--config", str(native_config),
                                             "--no-ui"], "audiocpp-backend.log", env)
                children.append(native)
                logs.append(log)
                wait_native(backend_url, native, "audiocpp-backend.log")
        facade = subprocess.Popen([sys.executable, str(ROOT / "server.py"), "--config",
                                   str(speech_config), *facade_args], cwd=ROOT, env=env,
                                  creationflags=WINDOWS_PROCESS_FLAGS)
        children.append(facade)
        while True:
            code = facade.poll()
            if code is not None:
                return code
            for child in children[:-1]:
                if child.poll() is not None:
                    raise RuntimeError("audio.cpp exited; see native backend log")
            time.sleep(0.25)
    finally:
        for child in reversed(children):
            stop_process(child)
        for log in logs:
            log.close()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"Cannot launch Castward Voices: {exc}")
