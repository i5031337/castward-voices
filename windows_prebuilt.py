"""Install a pinned audio.cpp Windows release without a compiler toolchain."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.error
import urllib.request
import uuid
import zipfile


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
PIN = ROOT / "windows_prebuilt.json"


def release_spec(cpu):
    pin = json.loads(PIN.read_text())
    kind = "cpu" if cpu else "cuda13.3"
    return pin, kind, pin["packages"][kind]


def runtime_path(pin, kind):
    return CACHE / "audio.cpp-prebuilt" / pin["version"] / kind


def binary_path(pin, kind):
    return runtime_path(pin, kind) / "audiocpp_server.exe"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(pin, item):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / item["file"]
    expected = item["sha256"].lower()
    if path.is_file() and sha256(path) == expected:
        print(f"Using verified archive: {path.name}", flush=True)
        return path
    partial = path.with_name(path.name + ".part")
    if partial.is_file() and sha256(partial) == expected:
        partial.replace(path)
        return path
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "Castward-Voices-Installer"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    url = pin["release_url"] + item["file"]
    print(f"Downloading {item['file']} ({'resuming' if offset else 'new'})", flush=True)
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not offset:
            raise
        partial.unlink()
        return download_archive(pin, item)
    with response:
        append = offset and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError(f"Unexpected download range for {item['file']}")
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    if sha256(partial) != expected:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch for {item['file']}; download was discarded")
    partial.replace(path)
    return path


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as data, target.open("wb") as output:
                    shutil.copyfileobj(data, output, length=1024 * 1024)


def remove_generated_dir(path):
    path = Path(path).resolve()
    allowed = (CACHE / "audio.cpp-prebuilt").resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError(f"Refusing to remove path outside the prebuilt cache: {path}")
    if path.exists():
        shutil.rmtree(path)


def read_manifest(runtime):
    try:
        return json.loads((runtime / "castward-install.json").read_text()), False
    except (FileNotFoundError, ValueError):
        return None, False
    except PermissionError:
        return None, True


def install(cpu):
    pin, kind, archives = release_spec(cpu)
    default_runtime = runtime_path(pin, kind)
    manifest = {"version": pin["version"], "kind": kind, "archives": archives}
    required = ["audiocpp_server.exe"]
    if not cpu:
        required += ["ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll",
                     "cublasLt64_13.dll", "cufft64_12.dll"]
    candidates = [default_runtime]
    if default_runtime.parent.is_dir():
        candidates.extend(sorted(default_runtime.parent.glob(f"{kind}-recovered-*")))
    default_inaccessible = False
    for candidate in candidates:
        installed, inaccessible = read_manifest(candidate)
        if candidate == default_runtime:
            default_inaccessible = inaccessible
        try:
            complete = installed == manifest and all((candidate / name).is_file() for name in required)
        except PermissionError:
            complete = False
        if complete:
            print(f"Using installed audio.cpp {pin['version']} ({kind})", flush=True)
            return candidate / "audiocpp_server.exe"
    runtime = (default_runtime.parent / f"{kind}-recovered-{uuid.uuid4().hex}"
               if default_inaccessible else default_runtime)
    binary = runtime / "audiocpp_server.exe"
    downloaded = [download_archive(pin, item) for item in archives]
    runtime.parent.mkdir(parents=True, exist_ok=True)
    # Path.mkdir inherits the cache ACL on Windows. tempfile.mkdtemp creates a
    # private directory that another account cannot read in a shared checkout.
    staged = runtime.parent / f".{kind}-{uuid.uuid4().hex}"
    staged.mkdir()
    try:
        for archive in downloaded:
            print(f"Extracting {archive.name}", flush=True)
            extract_archive(archive, staged)
        missing = [name for name in required if not (staged / name).is_file()]
        if missing:
            raise ValueError(f"Prebuilt archive is missing: {', '.join(missing)}")
        (staged / "castward-install.json").write_text(json.dumps(manifest, indent=2) + "\n")
        remove_generated_dir(runtime)
        os.replace(staged, runtime)
    finally:
        remove_generated_dir(staged)
    return binary
