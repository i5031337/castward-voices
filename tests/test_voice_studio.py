import base64
import io
import json
import re
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from server import create_app


CONFIG = {'default_model':'actor','alignment_model':None,'models':{
    'actor':{'backend_model':'voice','design_model':'designer','clone_model':'cloner',
             'instruction_option':'caption','reference_text_option':'transcript',
             'default_voice':'warm','voice_descriptions':{'warm':'Warm voice'},'max_input_chars':200},
    'reader':{'backend_model':'reader','context':True,'native_speed':True}}}


class NativeBackend:
    def __init__(self): self.calls=[]
    def discover(self):
        return [{'id':name,'family':'future_voice','params':[
            {'name':'future_guidance','type':'float','default':2.5},
            {'name':'use_reference','type':'bool','default':True},
            {'name':'style','type':'enum','values':['bright','calm']}]} for name in ['voice','designer','cloner','reader']]
    def voices(self,profile): return ['first','second']
    def generate(self,profile,body,**kwargs):
        self.calls.append((profile,body,kwargs))
        return np.sin(np.arange(44100)*0.1).astype(np.float32)*0.1,44100


class StudioTests(unittest.TestCase):
    def test_asset_url_changes_when_stylesheet_changes(self):
        page = self.client.get('/')
        self.assertEqual(page.headers['cache-control'], 'no-cache')
        asset_urls = lambda html: re.findall(r'/studio-assets/studio\.(?:css|js)\?v=[a-f0-9]+', html)
        before = asset_urls(page.text)
        self.assertEqual(len(before), 2)
        read_bytes = Path.read_bytes
        def changed(path):
            data = read_bytes(path)
            return data + b'\n/* updated theme */' if path.name == 'studio.css' else data
        with patch.object(Path, 'read_bytes', autospec=True, side_effect=changed):
            after = asset_urls(self.client.get('/').text)
        self.assertNotEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])
        self.assertEqual(self.client.get(before[0]).status_code, 200)

    def setUp(self):
        self.storage=tempfile.TemporaryDirectory()
        self.backend=NativeBackend()
        self.client=TestClient(create_app(config=CONFIG,backend=self.backend,voices_dir=self.storage.name,api_key='secret'))
        self.client.__enter__()
        self.auth={'Authorization':'Bearer secret'}
    def tearDown(self):
        self.client.__exit__(None,None,None);self.storage.cleanup()
    def generate(self,mode='design',**fields):
        speech={'model':'actor','input':'Hello from the studio.','instructions':'Warm and clear','response_format':'wav'}
        speech.update(fields.pop('speech',{}))
        return self.client.post('/ui/generate',headers=self.auth,json={'mode':mode,'speech':speech,**fields})
    def test_page_assets_auth_and_dynamic_model_metadata(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertIn('Castward Voice Studio',self.client.get('/').text)
        self.assertEqual(self.client.get('/studio-assets/studio.js').status_code,200)
        self.assertEqual(self.client.get('/ui/config').status_code,401)
        self.assertEqual(self.generate().status_code,200)
        self.assertEqual(self.client.post('/ui/generate',json={'mode':'speech','speech':{'model':'reader','input':'Hi'}}).status_code,401)
        config=self.client.get('/ui/config',headers=self.auth).json()
        self.assertEqual(config['native_models'][1]['params'][0]['name'],'future_guidance')
        self.assertEqual(config['profiles'][0]['design_model'],'designer')
        self.assertEqual(config['profiles'][0]['clone_model'],'cloner')
        self.assertNotIn('backend_api_key_env',config)
    def test_design_to_clone_preserves_native_recording_and_exact_transcript(self):
        result=self.generate(speech={'seed':str(2**64-1),'options':{'future_guidance':3,'use_reference':False}})
        self.assertEqual(result.status_code,200,result.text)
        take=result.json()
        preview,rate=sf.read(io.BytesIO(base64.b64decode(take['audio'])))
        self.assertEqual((len(preview),rate),(24000,24000))
        reference,rate=sf.read(io.BytesIO(base64.b64decode(take['reference_audio'])))
        self.assertEqual((len(reference),rate),(44100,44100))
        self.assertEqual(self.backend.calls[-1][1].seed,2**64-1)
        self.assertFalse(self.backend.calls[-1][1].options['use_reference'])
        clone=self.generate('clone',reference_audio=take['reference_audio'],reference_text=take['transcript'],
            speech={'input':'A new sentence.','instructions':None,'options':{'style':'calm'}})
        self.assertEqual(clone.status_code,200,clone.text)
        _,body,kwargs=self.backend.calls[-1]
        self.assertEqual(kwargs['mode'],'clone')
        self.assertEqual(kwargs['reference_text'],'Hello from the studio.')
        self.assertEqual(kwargs['reference'][1],44100)
        self.assertEqual(body.options,{'style':'calm'})
        self.assertEqual(list(Path(self.storage.name).iterdir()),[])
    def test_reference_upload_and_capability_validation_prevent_backend_calls(self):
        for params in [
            {'mode':'clone'},
            {'mode':'clone','reference_audio':'bad!!!'},
            {'mode':'clone','reference_audio':base64.b64encode(b'not audio').decode()},
            {'speech':{'model':'reader'}},
            {'speech':{'instructions':' '}},
            {'speech':{'input':'x'*201}},
            {'speech':{'npc_id':'saved'}},
            {'speech':{'stream':True}},
            {'reference_audio':'AAAA'},
        ]:
            response=self.generate(**params)
            self.assertEqual(response.status_code,400,response.text)
        self.assertEqual(self.backend.calls,[])
        self.assertEqual(self.generate('speech',speech={'model':'reader','instructions':None}).status_code,200)
    def test_silent_or_overlong_compressed_reference_rejected(self):
        for samples in (np.zeros(8000),np.zeros(8000*121)):
            stream=io.BytesIO();sf.write(stream,samples,8000,format='FLAC')
            r=self.generate('clone',reference_audio=base64.b64encode(stream.getvalue()).decode())
            self.assertEqual(r.status_code,400,r.text)
        self.assertEqual(self.backend.calls,[])
    def test_native_voice_option_can_override_configured_default(self):
        response=self.generate('speech',speech={'instructions':None,'options':{'voice_id':'other-preset'}})
        self.assertEqual(response.status_code,200,response.text)
        profile,_,kwargs=self.backend.calls[-1]
        self.assertIsNone(profile.default_voice)
        self.assertEqual(kwargs.get('mode'),'speech')
    def test_backend_failure_and_context_fallback(self):
        response=self.generate('speech',speech={'model':'reader','instructions':None,'context_before':'Before. '})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['context_status'],'fallback')
        def fail(*args,**kwargs): raise RuntimeError('native failure')
        self.backend.generate=fail
        self.assertEqual(self.generate().status_code,502)
