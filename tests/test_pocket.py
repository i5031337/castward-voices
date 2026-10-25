"""PocketTTS presets, transcript-free cloning, and reuse of saved NPC recordings."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest

import httpx
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from audiocpp_backend import AudioCppBackend
from backend_config import load_config
from launch_config import prepare_configs
from server import create_app, SpeechRequest

ROOT = Path(__file__).resolve().parents[1]


class PocketTests(unittest.TestCase):
    def test_presets_studio_clone_and_saved_npc_reuse(self):
        config = load_config(ROOT / 'backend-pocket.json')
        calls = []
        wave = io.BytesIO()
        sf.write(wave, .1 * np.sin(np.arange(96000) * .1), 24000, format='WAV')

        def handle(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'pocket-english', 'family': 'pocket_tts'}]})
            if request.url.path == '/v1/audio/voices':
                return httpx.Response(200, json={'voices': ['alba', 'anna']})
            calls.append(json.loads(request.content))
            return httpx.Response(200, content=wave.getvalue())

        backend = AudioCppBackend(config, transport=httpx.MockTransport(handle))
        with tempfile.TemporaryDirectory() as storage, TestClient(create_app(
                config=config, backend=backend, voices_dir=storage)) as client:
            profile = client.get('/ui/config').json()['profiles'][0]
            self.assertEqual(profile['voices'], ['alba', 'anna'])
            self.assertIsNone(profile['design_model'])
            self.assertIsNone(profile['reference_text_option'])
            body = {'model': 'pocket-tts', 'input': 'Welcome to the gate.', 'response_format': 'wav'}
            self.assertEqual(client.post('/v1/audio/speech', json=body).status_code, 200)
            self.assertEqual(calls[-1]['voice'], 'alba')
            response = client.post('/ui/generate', json={'mode': 'clone', 'speech': body,
                'reference_audio': base64.b64encode(wave.getvalue()).decode(), 'reference_text': 'Ignored transcript'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('voice_ref', calls[-1])
            self.assertNotIn('voice', calls[-1])
            self.assertEqual(calls[-1]['options'], {})
            # Existing references created with a design-capable profile can be reused.
            engine = client.app.state.engine
            engine.save_dialogue(SpeechRequest(**body, npc_id='guard'), 'pocket-tts', 'Saved voice', 42,
                                 .1 * np.sin(np.arange(96000) * .1), 24000)
            response = client.post('/v1/audio/speech', json={**body, 'npc_id': 'guard'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(client.get('/v1/voices/guard').json()['status'], 'ready')
            self.assertEqual(calls[-1]['options'], {})
            count = len(calls)
            for invalid in ({'npc_id': 'new', 'instructions': 'A guard'},
                            {'instructions': 'A guard'}, {'stream': True, 'response_format': 'pcm'},
                            {'seed': 4294967296}):
                self.assertEqual(client.post('/v1/audio/speech', json={**body, **invalid}).status_code, 400)
            self.assertEqual(len(calls), count)
        backend.close()

    def test_cpu_mode_routes_pocket_off_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            prepare_configs(ROOT / 'audiocpp-server.local.json', ROOT / 'backend.json', folder)
            gpu, cpu, api = [json.loads((Path(folder) / name).read_text())
                             for name in ('gpu.json', 'cpu.json', 'backend.json')]
            self.assertNotIn('pocket-english', [m['id'] for m in gpu['models']])
            pocket = next(m for m in cpu['models'] if m['id'] == 'pocket-english')
            self.assertNotIn('device', pocket)
            self.assertEqual(api['backend_routes']['pocket-english'], 'http://127.0.0.1:7863')
            self.assertEqual(gpu['max_loaded_models'], 1)
