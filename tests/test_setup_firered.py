"""Keep setup incremental and respect the user's native build targets."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FireRedSetupTests(unittest.TestCase):
    def run_setup(self, *, cached=False, cpu=False, targets=None, custom=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'audio.cpp'
            session = source / 'src/models/fireredtts3/session.cpp'
            session.parent.mkdir(parents=True)
            session.touch()
            build = root / 'custom-build' if custom else source / 'build/ming-cuda'
            if cached:
                build.mkdir(parents=True)
                (build / 'CMakeCache.txt').write_text('CMAKE_CUDA_ARCHITECTURES:STRING=86-real\n')
            commands = root / 'commands'
            commands.mkdir()
            log = root / 'calls.jsonl'
            # Stub only external commands: no compilation, downloads, or source writes.
            for name in ('cmake', 'hf'):
                executable = commands / name
                executable.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'with open(os.environ["SETUP_TEST_LOG"],"a") as log:\n'
                    '    log.write(json.dumps(sys.argv)+"\\n")\n')
                executable.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in (
                'AUDIOCPP_BUILD_DIR', 'CUDA_ARCHITECTURES', 'BUILD_JOBS')}
            env.update(AUDIOCPP_SOURCE=str(source), SETUP_TEST_LOG=str(log),
                       PATH=str(commands) + os.pathsep + env['PATH'])
            if custom:
                env['AUDIOCPP_BUILD_DIR'] = str(build)
            if targets:
                env['CUDA_ARCHITECTURES'] = targets
            subprocess.run(['bash', str(ROOT / 'setup-firered.sh'), *(['--cpu'] if cpu else [])],
                           env=env, check=True, capture_output=True, text=True)
            return [json.loads(line)[1:] for line in log.read_text().splitlines()], str(build)

    def test_cached_build_keeps_compilers_targets_and_build_type(self):
        calls, build = self.run_setup(cached=True)
        configure, compile, download = calls
        self.assertEqual(configure[configure.index('-B') + 1], build)
        self.assertFalse(any(arg.startswith(('-DCMAKE_CUDA_ARCHITECTURES=', '-DCMAKE_CUDA_COMPILER=',
            '-DCMAKE_CXX_COMPILER=', '-DCMAKE_BUILD_TYPE=', '-DAUDIOCPP_DEPLOYMENT_BUILD=')) for arg in configure))
        self.assertIn('-DAUDIOCPP_MODELS=ming3_tts,kokoro_tts,qwen3_tts,fireredtts3', configure)
        self.assertEqual(compile, ['--build', build, '--target', 'audiocpp_server', '--parallel', '2'])
        self.assertIn('ef1ab2598f6d4d101486e4c5b9e605c687f20642', download)

    def test_fresh_build_detects_local_gpus_and_accepts_other_targets(self):
        calls, _ = self.run_setup()
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=native', calls[0])
        calls, build = self.run_setup(cached=True, targets='75-real;86-real', custom=True)
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real', calls[0])
        self.assertEqual(calls[0][calls[0].index('-B') + 1], build)

    def test_cpu_build_uses_separate_directory_without_cuda_targets(self):
        calls, _ = self.run_setup(cpu=True)
        configure = calls[0]
        self.assertEqual(configure[configure.index('-B') + 1], str(ROOT / '.cache/audio.cpp-cpu-build'))
        self.assertIn('-DENGINE_ENABLE_CUDA=OFF', configure)
        self.assertFalse(any(arg.startswith('-DCMAKE_CUDA_ARCHITECTURES=') for arg in configure))
