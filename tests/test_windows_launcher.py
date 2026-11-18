"""Check Windows process ownership without starting model inference."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_local_windows as launcher


class FakeProcess:
    def __init__(self, *, exit_on_poll=False):
        self.exit_on_poll = exit_on_poll
        self.stopped = False

    def poll(self):
        return 0 if self.stopped or self.exit_on_poll else None

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0


class WindowsLauncherTests(unittest.TestCase):
    def test_split_mode_starts_both_backends_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'audiocpp_server.exe'
            binary.touch()
            native = root / 'native.json'
            speech = root / 'backend.json'
            native.write_text('{}')
            speech.write_text('{}')
            started = []
            facade = FakeProcess(exit_on_poll=True)

            def fake_start(command, logfile, env):
                process = FakeProcess()
                started.append((command, process))
                return process, io.StringIO()

            def fake_prepare(native_path, speech_path, output_dir, **kwargs):
                (output_dir / 'backend.json').write_text('{}')

            with patch.object(launcher, 'ROOT', root), \
                    patch.object(launcher, 'CACHE', root / '.cache'), \
                    patch.object(launcher, 'installed_paths', return_value=(binary, native, speech)), \
                    patch.object(launcher, 'prepare_configs', side_effect=fake_prepare) as prepare, \
                    patch.object(launcher, 'check_ports') as ports, \
                    patch.object(launcher, 'start_process', side_effect=fake_start), \
                    patch.object(launcher, 'wait_native'), \
                    patch.object(launcher.subprocess, 'Popen', return_value=facade) as popen:
                self.assertEqual(launcher.run(['--kokoro-cpu', '--gpu-device', '2', '--port', '8002']), 0)
            self.assertEqual(len(started), 2)
            self.assertTrue(all(process.stopped for _, process in started))
            self.assertEqual(prepare.call_args.kwargs['gpu_device'], 2)
            ports.assert_called_once_with(7864, 7863)
            self.assertEqual(popen.call_args.args[0][-2:], ['--port', '8002'])
            self.assertEqual(list((root / '.cache').glob('launch.*')), [])

    def test_plain_mode_reuses_healthy_native_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / 'native.json'
            speech = root / 'backend.json'
            native.write_text('{}')
            speech.write_text(json.dumps({'backend_url': 'http://127.0.0.1:7862'}))
            with patch.object(launcher, 'ROOT', root), \
                    patch.object(launcher, 'CACHE', root / '.cache'), \
                    patch.object(launcher, 'installed_paths', return_value=(root / 'missing.exe', native, speech)), \
                    patch.object(launcher, 'healthy', return_value=True), \
                    patch.object(launcher, 'start_process') as native_start, \
                    patch.object(launcher.subprocess, 'Popen', return_value=FakeProcess(exit_on_poll=True)):
                self.assertEqual(launcher.run([]), 0)
            native_start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
