import copy
from pathlib import Path
import tempfile
import unittest
import httpx
from fastapi.testclient import TestClient
from audiocpp_backend import AudioCppBackend
from backend_config import load_config, ServerConfig
from server import SpeechRequest, create_app
from test_voice_studio import NativeBackend


class OptionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings=load_config(Path(__file__).resolve().parents[1]/'backend.json')
        self.profile=self.settings.models['npc']
    def test_invalid_template_controls_are_rejected_before_native_http(self):
        calls=[]
        backend=AudioCppBackend(self.settings,transport=httpx.MockTransport(lambda req:calls.append(req)))
        for name,value in [('voice_id','US_Emma'),('speed','快速'),('pitch','高'),('volume','高'),('emotion','happy'),('dialect','cantonese')]:
            body=SpeechRequest(model='npc',input='Hello',instructions='A voice',options={name:value})
            with self.assertRaisesRegex(ValueError,name):
                backend.generate(self.profile,body,mode='design',instruction=body.instructions)
        self.assertEqual(calls,[])
        backend.close()
    def test_ming_native_presets_and_explicit_template_options(self):
        backend=AudioCppBackend(self.settings)
        body=SpeechRequest(model='npc',input='Hello',voice='US_Emma')
        payload=backend.payload(self.profile,body)
        self.assertEqual(payload['options']['template_name'],'ip')
        self.assertEqual(payload['options']['voice_id'],'US_Emma')
        self.assertNotIn('voice',payload)
        body=SpeechRequest(model='npc',input='Hello',options={'template_name':'ip','voice_id':'UK_George'})
        self.assertEqual(backend.payload(self.profile,body,mode='design')['options']['voice_id'],'UK_George')
        with self.assertRaisesRegex(ValueError,'structured controls'):
            backend.payload(self.profile,body,mode='design',instruction='A new voice')
        body=SpeechRequest(model='npc',input='Hello',options={'template_name':'basic','pitch':'高'})
        opts=backend.payload(self.profile,body,mode='design')['options']
        self.assertNotIn('voice_id',opts)
        self.assertEqual(opts['pitch'],'高')
        backend.close()
    def test_studio_returns_actionable_400_and_accepts_structured_templates(self):
        profile=self.profile.model_copy(update={'backend_model':'voice','design_model':'designer','clone_model':'cloner'})
        config={'default_model':'actor','alignment_model':None,'models':{'actor':profile.model_dump()}}
        backend=NativeBackend()
        with tempfile.TemporaryDirectory() as directory, TestClient(create_app(config=config,backend=backend,voices_dir=directory)) as client:
            response=client.post('/ui/generate',json={'mode':'design','speech':{'model':'actor','input':'Hello','instructions':'A voice','options':{'voice_id':'US_Emma'}}})
            self.assertEqual(response.status_code,400,response.text)
            self.assertIn('template_name=ip',response.json()['error']['message'])
            self.assertEqual(backend.calls,[])
            response=client.post('/ui/generate',json={'mode':'design','speech':{'model':'actor','input':'Hello','options':{'template_name':'ip','voice_id':'US_Emma'}}})
            self.assertEqual(response.status_code,200,response.text)
            self.assertIsNone(backend.calls[-1][2].get('instruction'))
    def test_qwen_labels_distinguish_designer_cloner_and_custom_voices(self):
        small=self.settings.models['qwen3-tts'];large=self.settings.models['qwen3-tts-1.7b'];custom=self.settings.models['qwen3-customvoice']
        self.assertEqual(small.design_model,large.design_model)
        self.assertNotEqual(small.clone_model,large.clone_model)
        self.assertIn('clone 0.6B',small.display_name)
        self.assertIn('clone 1.7B',large.display_name)
        self.assertIn('CustomVoice',custom.display_name)
        self.assertIsNone(custom.clone_model)
