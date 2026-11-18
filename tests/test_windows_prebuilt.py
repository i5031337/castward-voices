"""Verify pinned archive installation without network or model downloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import windows_prebuilt as prebuilt


class WindowsPrebuiltTests(unittest.TestCase):
    def test_cpu_and_cuda_archives_extract_into_separate_runtime_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            files = {}
            for name, members in {
                    'cpu.zip': {'audiocpp_server.exe': b'cpu'},
                    'cuda.zip': {'audiocpp_server.exe': b'cuda', 'ggml-cuda.dll': b'gpu'},
                    'runtime.zip': {name: b'dll' for name in (
                        'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll',
                        'cufft64_12.dll')},
            }.items():
                path = cache / name
                with zipfile.ZipFile(path, 'w') as archive:
                    for member, content in members.items():
                        archive.writestr(member, content)
                files[name] = {'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            pin = {'version': 'vtest', 'release_url': 'https://example.invalid/', 'packages': {
                'cpu': [files['cpu.zip']], 'cuda13.3': [files['cuda.zip'], files['runtime.zip']]}}
            pin_path = cache / 'pin.json'
            pin_path.write_text(json.dumps(pin))
            with patch.object(prebuilt, 'CACHE', cache), patch.object(prebuilt, 'PIN', pin_path):
                cpu = prebuilt.install(True)
                cuda = prebuilt.install(False)
                self.assertEqual(cpu.read_bytes(), b'cpu')
                self.assertEqual(cuda.read_bytes(), b'cuda')
                self.assertTrue((cuda.parent / 'cudart64_13.dll').is_file())
                self.assertEqual(prebuilt.install(False), cuda)

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('../outside.txt', 'unsafe')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                prebuilt.extract_archive(archive, root / 'runtime')
            self.assertFalse((root / 'outside.txt').exists())

    def test_unreadable_existing_runtime_uses_recoverable_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            archive = cache / 'cpu.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('audiocpp_server.exe', b'cpu')
            item = {'file': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            pin = {'version': 'vtest', 'release_url': 'https://example.invalid/',
                   'packages': {'cpu': [item], 'cuda13.3': []}}
            pin_path = cache / 'pin.json'
            pin_path.write_text(json.dumps(pin))
            original_read = prebuilt.read_manifest

            def simulate_permissions(runtime):
                if runtime == prebuilt.runtime_path(pin, 'cpu'):
                    return None, True
                return original_read(runtime)

            with patch.object(prebuilt, 'CACHE', cache), patch.object(prebuilt, 'PIN', pin_path), \
                    patch.object(prebuilt, 'read_manifest', side_effect=simulate_permissions):
                binary = prebuilt.install(True)
                self.assertTrue(binary.is_file())
                self.assertTrue(binary.parent.name.startswith('cpu-recovered-'))
                self.assertEqual(prebuilt.install(True), binary)


if __name__ == '__main__':
    unittest.main()
