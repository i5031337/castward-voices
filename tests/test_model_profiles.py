import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import httpx
from fastapi.testclient import TestClient
from audiocpp_backend import AudioCppBackend
from backend_config import load_config, ServerConfig
from server import create_app, SpeechRequest
from test_server import FakeBackend, CONFIG


class ModelProfileTests(unittest.TestCase):
    def test_public_models_show_presets_and_every_designer_cloner_pair(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'backend.json')
        public = config.public_models()
        designers = {'ming', 'qwen-design-1.7b', 'firered-instruct'}
        cloners = {'ming', 'qwen-base-0.6b', 'qwen-base-1.7b', 'firered-instruct', 'pocket-english'}
        pairs = {f'design-{designer}--clone-{cloner}' for designer in designers for cloner in cloners}
        self.assertEqual(set(public), pairs | {'ming-presets', 'narrator', 'qwen3-customvoice', 'pocket-tts', 'fireredtts3-presets'})
        self.assertEqual(config.public_default_model(), 'design-ming--clone-ming')
        self.assertTrue(all(not public[name].preset_enabled for name in pairs))
        self.assertTrue(all(public[name].preset_enabled for name in set(public) - pairs))

        class DiscoveryBackend:
            def discover(self):
                native_ids = {value for profile in config.models.values() for value in
                              (profile.backend_model, profile.design_model, profile.clone_model,
                               profile.streaming_model) if value}
                return [{'id': model} for model in native_ids]
            def voices(self, profile):
                return []

        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(
                config=config, backend=DiscoveryBackend(), voices_dir=directory)) as client:
            models = client.get('/v1/models').json()['data']
            self.assertEqual({entry['id'] for entry in models}, set(public))
            self.assertFalse({*config.aliases, 'npc', 'qwen3-tts'} & {entry['id'] for entry in models})
            studio = client.get('/ui/config').json()
            self.assertEqual({entry['id'] for entry in studio['profiles']}, set(public))
            self.assertEqual(studio['default_model'], config.public_default_model())

    def test_cross_family_pairs_keep_each_models_native_options(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'backend.json')
        backend = AudioCppBackend(config)
        try:
            reference = (np.ones(2400) * 0.1, 24000)
            body = SpeechRequest(model='design-firered-instruct--clone-pocket-english',
                                 input='The gate is open.', language='English', seed=42)
            profile = config.resolve(body.model)[1]
            design = backend.payload(profile, body, mode='design', instruction='An older guard')
            clone = backend.payload(profile, body, mode='clone', reference=reference,
                                    reference_text='The gate is open.')
            self.assertEqual(design['model'], 'firered-instruct')
            self.assertEqual(design['options']['template_name'], 'voice_design')
            self.assertEqual(design['options']['language'], 'English')
            self.assertNotIn('max_tokens', design['options'])
            self.assertEqual(clone['model'], 'pocket-english')
            self.assertEqual(clone['options'], {})
            self.assertNotIn('voice', clone)
            body = body.model_copy(update={'model': 'design-ming--clone-qwen-base-0.6b'})
            profile = config.resolve(body.model)[1]
            design = backend.payload(profile, body, mode='design', instruction='An older guard')
            clone = backend.payload(profile, body, mode='clone', reference=reference,
                                    reference_text='The gate is open.')
            self.assertEqual((design['model'], design['options']['template_name']), ('ming', 'voice_design'))
            self.assertEqual(clone['model'], 'qwen-base-0.6b')
            self.assertNotIn('template_name', clone['options'])
        finally:
            backend.close()

    def test_studio_generates_with_cross_family_pair(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'backend.json')
        calls = []
        output = io.BytesIO()
        sf.write(output, np.ones(24000) * 0.1, 24000, format='WAV')
        native_ids = {value for profile in config.models.values() for value in
                      (profile.backend_model, profile.design_model, profile.clone_model,
                       profile.streaming_model) if value}

        def handle(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': name} for name in native_ids]})
            if request.url.path == '/v1/audio/voices':
                return httpx.Response(200, json={'voices': []})
            calls.append(json.loads(request.content))
            return httpx.Response(200, content=output.getvalue())

        backend = AudioCppBackend(config, transport=httpx.MockTransport(handle))
        try:
            with tempfile.TemporaryDirectory() as directory, TestClient(create_app(
                    config=config, backend=backend, voices_dir=directory)) as client:
                model = 'design-firered-instruct--clone-pocket-english'
                design = client.post('/ui/generate', json={'mode': 'design', 'speech': {
                    'model': model, 'input': 'Hello there.', 'instructions': 'A clear, calm voice',
                    'response_format': 'wav'}})
                self.assertEqual(design.status_code, 200, design.text)
                self.assertEqual(calls[-1]['model'], 'firered-instruct')
                self.assertEqual(calls[-1]['options']['template_name'], 'voice_design')
                clone = client.post('/ui/generate', json={'mode': 'clone',
                    'reference_audio': design.json()['reference_audio'],
                    'reference_text': 'Hello there.', 'speech': {
                        'model': model, 'input': 'Welcome back.', 'response_format': 'wav'}})
                self.assertEqual(clone.status_code, 200, clone.text)
                self.assertEqual(calls[-1]['model'], 'pocket-english')
                self.assertEqual(calls[-1]['options'], {})
                self.assertIn('voice_ref', calls[-1])
                preset = client.post('/v1/audio/speech', json={
                    'model': 'ming-presets', 'input': 'Welcome back.', 'response_format': 'wav'})
                self.assertEqual(preset.status_code, 200, preset.text)
                self.assertEqual(calls[-1]['model'], 'ming')
                self.assertEqual(calls[-1]['options']['template_name'], 'ip')
                self.assertEqual(calls[-1]['options']['voice_id'], 'US_Emma')
                self.assertNotIn('instruct', calls[-1]['options'])
        finally:
            backend.close()

    def test_configured_qwen_routes_and_compatibility_aliases(self):
        config=load_config(Path(__file__).resolve().parents[1]/'backend.json')
        self.assertEqual(config.models['narrator'].display_name,'Kokoro 82M · preset voices')
        self.assertEqual(config.resolve('qwen3-tts')[1].clone_model,'qwen-base-0.6b')
        self.assertEqual(config.resolve('qwen3-tts-0.6b')[0],'qwen3-tts')
        self.assertEqual(config.resolve('qwen3-tts-1.7b')[1].clone_model,'qwen-base-1.7b')
        self.assertEqual(config.resolve('qwen3-tts')[1].design_model,'qwen-design-1.7b')
        self.assertEqual(config.resolve('npc')[1].clone_model,'ming')
        self.assertIsNone(config.resolve('qwen3-customvoice')[1].clone_model)

    def test_firered_pilot_maps_design_and_clone_templates(self):
        config=load_config(Path(__file__).resolve().parents[1]/'backend-firered.json')
        profile=config.models['fireredtts3']
        backend=AudioCppBackend(config)
        body=SpeechRequest(model='fireredtts3',input='The gate is open.',language='English',seed=42,
                           max_new_tokens=128)
        design=backend.payload(profile,body,mode='design',instruction='An older, gravelly guard')
        self.assertEqual(design['model'],'firered-instruct')
        self.assertEqual(design['options']['template_name'],'voice_design')
        self.assertEqual(design['options']['instruct'],'An older, gravelly guard')
        self.assertEqual(design['options']['language'],'English')
        self.assertNotIn('max_tokens',design['options'])
        reference=(np.ones(2400)*0.1,24000)
        clone=backend.payload(profile,body,mode='clone',reference=reference,reference_text='The gate is open.')
        self.assertEqual(clone['options']['template_name'],'instruct_tts')
        self.assertEqual(clone['options']['reference_text'],'The gate is open.')
        self.assertNotIn('max_tokens',clone['options'])
        self.assertIn('voice_ref',clone)
        backend.close()

    def test_speaker_only_is_opt_in_and_explicit_overrides_remain_available(self):
        config=ServerConfig.model_validate(CONFIG)
        backend=AudioCppBackend(config)
        profile=config.models['actor'].model_copy(update={'speaker_only_option':'speaker_only','seed_max':2**32-1})
        reference=(np.ones(2400)*0.1,24000)
        body=SpeechRequest(model='actor',input='Hello',seed=2**32-1)
        request=backend.payload(profile,body,mode='clone',reference=reference)
        self.assertIs(request['options']['speaker_only'],True)
        self.assertEqual(request['seed'],str(2**32-1))
        with_text=backend.payload(profile,body,mode='clone',reference=reference,reference_text='Exact words')
        self.assertNotIn('speaker_only',with_text['options'])
        self.assertEqual(with_text['options']['reference_text'],'Exact words')
        explicit=body.model_copy(update={'options':{'speaker_only':False}})
        self.assertIs(backend.payload(profile,explicit,mode='clone',reference=reference)['options']['speaker_only'],False)
        generic=backend.payload(config.models['actor'],body,mode='clone',reference=reference)
        self.assertNotIn('speaker_only',generic['options'])
        with self.assertRaisesRegex(ValueError,'4294967295'):
            backend.payload(profile,body.model_copy(update={'seed':2**32}),mode='design')
        backend.close()

    def test_npc_derived_and_legacy_reference_extension_seeds_fit_profile_limit(self):
        config=copy.deepcopy(CONFIG)
        config['models']['actor']['seed_max']=2**32-1
        backend=FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            app=create_app(config=config,backend=backend,voices_dir=directory)
            with TestClient(app) as client:
                response=client.post('/v1/audio/speech',json={'model':'actor','npc_id':'qwen-seed-check','instructions':'A voice','input':'short','response_format':'wav'})
                self.assertEqual(response.status_code,200,response.text)
                folder=Path(directory)/hashlib.sha256(b'qwen-seed-check').hexdigest()
                metadata=json.loads((folder/'metadata.json').read_text())
                self.assertLessEqual(metadata['seed'],2**32-1)
                metadata.update(seed=2**60+11,status='failed')
                (folder/'metadata.json').write_text(json.dumps(metadata))
                (folder/'reference.wav').unlink()
                app.state.engine.prepare('qwen-seed-check')
                self.assertEqual(backend.calls[-1][0].seed,(2**60+11)%(2**32))
                public=client.get('/ui/config').json()['profiles'][0]
                self.assertEqual(public['seed_max'],'4294967295')
