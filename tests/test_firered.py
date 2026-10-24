"""FireRed through the public API and Studio, using the real HTTP adapter."""
import base64
import hashlib
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
from server import create_app

ROOT = Path(__file__).resolve().parents[1]


class FireRedTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'backend.json')
        self.calls = []
        self.storage = tempfile.TemporaryDirectory()
        native = json.loads((ROOT / 'audiocpp-server.local.json').read_text())
        self.native = native

        def native_http(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [
                    {'id': model['id'], 'family': model['family'], 'params': [
                        {'name': 'num_inference_steps', 'type': 'int', 'default': 10},
                        {'name': 'template_name', 'type': 'enum', 'values': [
                            'voice_design', 'instruct_tts', 'semantic_edit', 'acoustic_edit']}]} for model in native['models']]})
            if request.url.path == '/v1/audio/voices':
                return httpx.Response(200, json={'voices': []})
            payload = json.loads(request.content)
            self.calls.append(payload)
            seconds = 1 if payload['input'] == 'Hi.' else 5
            audio = np.sin(np.arange(seconds * 24000) * .1) * .1
            stream = io.BytesIO()
            sf.write(stream, audio, 24000, format='WAV')
            return httpx.Response(200, content=stream.getvalue())

        self.backend = AudioCppBackend(self.config, transport=httpx.MockTransport(native_http))
        self.client = TestClient(create_app(config=self.config, backend=self.backend, voices_dir=self.storage.name))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.backend.close()
        self.storage.cleanup()

    def speech(self, **overrides):
        return self.client.post('/v1/audio/speech', json={
            'model': 'fireredtts3', 'input': 'Welcome to the gate.', 'response_format': 'wav', **overrides})

    def test_standard_discovery_presets_and_language_mapping(self):
        self.assertIn('fireredtts3', [entry['id'] for entry in self.client.get('/v1/models').json()['data']])
        voices = self.client.get('/v1/voices').json()['data']
        self.assertIn({'model': 'fireredtts3', 'id': 'echo'}, voices)
        config = self.client.get('/ui/config').json()
        profile = next(p for p in config['profiles'] if p['id'] == 'fireredtts3')
        self.assertEqual(profile['seed_max'], '4294967295')
        self.assertIsNone(profile['streaming_model'])
        self.assertEqual(profile['language_map']['Turkish'], 'Turkish')
        self.assertEqual(self.speech(voice='echo', language='auto').status_code, 200)
        payload = self.calls[-1]
        self.assertEqual(payload['model'], 'firered-instruct')
        self.assertEqual(payload['options']['language'], 'English')
        self.assertEqual(payload['options']['template_name'], 'voice_design')
        self.assertIn('male', payload['options']['instruct'])
        self.assertNotIn('voice', payload)

    def test_studio_design_clone_options_and_invalid_templates(self):
        response = self.client.post('/ui/generate', json={'mode': 'design', 'speech': {
            'model': 'fireredtts3', 'input': 'Welcome to the gate.', 'instructions': 'A gravelly guard',
            'seed': 4294967295, 'options': {'num_inference_steps': 4, 'guidance_scale': 1.5}}})
        self.assertEqual(response.status_code, 200, response.text)
        take = response.json()
        self.assertEqual(self.calls[-1]['seed'], '4294967295')
        self.assertEqual(self.calls[-1]['options']['num_inference_steps'], 4)
        clone = {'mode': 'clone', 'reference_audio': take['reference_audio'], 'reference_text': take['transcript'],
                 'speech': {'model': 'fireredtts3', 'input': 'You may pass.', 'options': {'stop_threshold': .4}}}
        response = self.client.post('/ui/generate', json=clone)
        self.assertEqual(response.status_code, 200, response.text)
        payload = self.calls[-1]
        self.assertEqual(payload['options']['template_name'], 'instruct_tts')
        self.assertEqual(payload['options']['reference_text'], take['transcript'])
        self.assertEqual(base64.b64decode(payload['voice_ref']['data'])[:4], b'RIFF')
        self.assertNotIn('instruct', payload['options'])
        count = len(self.calls)
        for template in ('semantic_edit', 'acoustic_edit', 'voice_design'):
            clone['speech']['options'] = {'template_name': template}
            response = self.client.post('/ui/generate', json=clone)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn('instruct_tts', response.text)
        self.assertEqual(self.speech(seed=4294967296).status_code, 400)
        self.assertEqual(self.speech(stream=True, response_format='pcm').status_code, 400)
        self.assertEqual(len(self.calls), count)

    def test_short_npc_reference_extension_and_persistent_reuse(self):
        response = self.speech(npc_id='firered-guard', input='Hi.', instructions='An older guard')
        self.assertEqual(response.status_code, 200, response.text)
        design, extension = self.calls
        self.assertEqual(design['options']['template_name'], 'voice_design')
        self.assertEqual(extension['options']['template_name'], 'instruct_tts')
        self.assertEqual(extension['options']['reference_text'], 'Hi.')
        folder = Path(self.storage.name) / hashlib.sha256(b'firered-guard').hexdigest()
        metadata = json.loads((folder / 'metadata.json').read_text())
        self.assertEqual(metadata['status'], 'ready')
        self.assertEqual(metadata['model'], 'fireredtts3')
        original = (folder / 'reference.wav').read_bytes()
        response = self.speech(npc_id='firered-guard')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.calls[-1]['options']['reference_text'], metadata['ref_text'])
        self.assertEqual(original, (folder / 'reference.wav').read_bytes())
        self.assertEqual(len(self.calls), 3)

    def test_cpu_narration_routes_firered_to_single_resident_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            prepare_configs(ROOT / 'audiocpp-server.local.json', ROOT / 'backend.json', folder)
            gpu = json.loads((Path(folder) / 'gpu.json').read_text())
            cpu = json.loads((Path(folder) / 'cpu.json').read_text())
            self.assertEqual(gpu['max_loaded_models'], 1)
            model = next(m for m in gpu['models'] if m['id'] == 'firered-instruct')
            self.assertEqual(model['device'], 0)
            self.assertEqual(model['path'], str(ROOT / self.native['models'][-1]['path']))
            self.assertNotIn('firered-instruct', [m['id'] for m in cpu['models']])
