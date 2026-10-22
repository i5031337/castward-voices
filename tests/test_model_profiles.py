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
from fastapi.testclient import TestClient
from audiocpp_backend import AudioCppBackend
from backend_config import load_config, ServerConfig
from server import create_app, SpeechRequest
from test_server import FakeBackend, CONFIG


class ModelProfileTests(unittest.TestCase):
    def test_configured_qwen_routes_and_compatibility_aliases(self):
        config=load_config(Path(__file__).resolve().parents[1]/'backend.json')
        self.assertEqual(config.resolve('qwen3-tts')[1].clone_model,'qwen-base-0.6b')
        self.assertEqual(config.resolve('qwen3-tts-0.6b')[0],'qwen3-tts')
        self.assertEqual(config.resolve('qwen3-tts-1.7b')[1].clone_model,'qwen-base-1.7b')
        self.assertEqual(config.resolve('qwen3-tts')[1].design_model,'qwen-design-1.7b')
        self.assertEqual(config.resolve('npc')[1].clone_model,'ming')
        self.assertIsNone(config.resolve('qwen3-customvoice')[1].clone_model)

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
