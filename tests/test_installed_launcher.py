"""Launch setup-generated configs through the real shell entry point, without inference."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstalledLauncherTests(unittest.TestCase):
    def test_installed_binary_and_configs_are_used_and_child_is_cleaned_up(self):
        with tempfile.TemporaryDirectory(prefix='castward launch ') as directory:
            root = Path(directory)
            shutil.copy(ROOT / 'run-local.sh', root)
            install = root / '.cache/install'
            install.mkdir(parents=True)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            binary = root / 'native server'
            binary.write_text('''#!/usr/bin/env python3
import json,os,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
config=Path(sys.argv[sys.argv.index('--config')+1])
settings=json.loads(config.read_text())
Path('native-selection.json').write_text(json.dumps({'config':str(config),'pid':os.getpid()}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok","backend":"cuda"}')
    def log_message(self,*args): pass
HTTPServer(('127.0.0.1',settings['port']),Handler).serve_forever()
''')
            binary.chmod(0o755)
            python = root / 'python wrapper'
            python.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
if sys.argv[1]=='server.py':
    Path('facade-selection.json').write_text(json.dumps(sys.argv[2:]))
else:
    os.execv(os.environ['REAL_TEST_PYTHON'],[os.environ['REAL_TEST_PYTHON'],*sys.argv[1:]])
''')
            python.chmod(0o755)
            (install / 'settings.json').write_text('{}')
            (install / 'server-path').write_text(str(binary)+'\n')
            (install / 'native.json').write_text(json.dumps({'port':port}))
            (install / 'backend.json').write_text(json.dumps({'backend_url':f'http://127.0.0.1:{port}'}))
            env = {k:v for k,v in os.environ.items() if k not in (
                'AUDIOCPP_SERVER','AUDIOCPP_CONFIG','SPEECH_CONFIG')}
            env.update(PYTHON=str(python), REAL_TEST_PYTHON=str(ROOT / '.venv/bin/python'))
            subprocess.run(['bash', str(root / 'run-local.sh')], env=env, cwd=root,
                           timeout=20, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((root / 'facade-selection.json').read_text()),
                             ['--config',str(install / 'backend.json')])
            native = json.loads((root / 'native-selection.json').read_text())
            self.assertEqual(native['config'],str(install / 'native.json'))
            with self.assertRaises(ProcessLookupError):
                os.kill(native['pid'],0)


if __name__ == '__main__':
    unittest.main()
