"""Shared native builds retain cached settings; compatibility helpers delegate."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from setup_models import build_commands, read_json

ROOT = Path(__file__).resolve().parents[1]


class FireRedSetupTests(unittest.TestCase):
    def test_cached_build_keeps_compilers_targets_and_build_type(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / 'CMakeCache.txt').touch()
            configure, compile = build_commands(ROOT, build, read_json(ROOT / 'model_catalog.json'), cpu=False, jobs=2)
            self.assertFalse(any(arg.startswith(('-DCMAKE_CUDA_ARCHITECTURES=', '-DCMAKE_CUDA_COMPILER=',
                '-DCMAKE_CXX_COMPILER=', '-DCMAKE_BUILD_TYPE=', '-DAUDIOCPP_DEPLOYMENT_BUILD=')) for arg in configure))
            self.assertIn('-DAUDIOCPP_MODELS=ming3_tts,kokoro_tts,qwen3_tts,fireredtts3,pocket_tts', configure)
            self.assertEqual(compile, ['cmake', '--build', str(build), '--target', 'audiocpp_server', '--parallel', '2'])

    def test_fresh_cpu_build_omits_cuda_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            configure, _ = build_commands(ROOT, Path(directory), read_json(ROOT / 'model_catalog.json'), cpu=True, jobs=2)
            self.assertIn('-DENGINE_ENABLE_CUDA=OFF', configure)
            self.assertFalse(any(arg.startswith('-DCMAKE_CUDA_ARCHITECTURES=') for arg in configure))

    def test_model_helpers_delegate_to_shared_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'uv'
            stub.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                            'open(os.environ["SETUP_TEST_LOG"],"w").write(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            log = root / 'log'
            environment = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'], 'SETUP_TEST_LOG': str(log)}
            for name in ('firered', 'pocket'):
                subprocess.run(['bash', str(ROOT / f'setup-{name}.sh'), '--cpu'], env=environment, check=True)
                self.assertEqual(json.loads(log.read_text())[-3:], ['--add', name, '--cpu'])


if __name__ == '__main__':
    unittest.main()
