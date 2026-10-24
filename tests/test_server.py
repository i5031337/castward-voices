import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from server import create_app


CONFIG={'default_model':'actor','aliases':{'tts-1':'actor'},'alignment_model':None,'models':{
    'actor':{'backend_model':'native','design_model':'native','clone_model':'native','instruction_option':'caption',
             'default_voice':'warm','voice_descriptions':{'warm':'Warm voice'}},
    'reader':{'backend_model':'reader','streaming_model':'stream','streaming_sample_rate':24000,'native_speed':True,
              'context':True,'voice_aliases':{'george':'british'}}}}


class FakeBackend:
    def __init__(self):
        self.calls=[];self.closed=False;self.stream_closed=False;self.fail=False
    def discover(self):
        return [{'id':name} for name in ('native','reader','stream')]
    def voices(self,profile):
        return ['british','american']
    def generate(self,profile,body,**kwargs):
        self.calls.append((body,kwargs))
        if self.fail:
            raise RuntimeError('failed')
        samples=22050 if body.input=='short' and kwargs.get('mode')=='design' else 176400
        return np.sin(np.arange(samples)*0.1).astype(np.float32)*0.1,44100
    def stream(self,profile,body):
        try:
            yield np.ones(2400,dtype=np.float32)*0.1
            yield np.ones(4800,dtype=np.float32)*0.2
        finally:
            self.stream_closed=True
    def close(self): self.closed=True


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.storage=tempfile.TemporaryDirectory();self.backend=FakeBackend()
        self.start()
    def start(self):
        self.client=TestClient(create_app(config=CONFIG,backend=self.backend,voices_dir=self.storage.name,api_key='key'))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None);self.storage.cleanup()
    def request(self,**kwargs):
        body={'model':'actor','input':'Hello','response_format':'wav'};body.update(kwargs)
        return self.client.post('/v1/audio/speech',json=body,headers={'Authorization':'Bearer key'})
    def folder(self,npc='guard'):
        return Path(self.storage.name)/hashlib.sha256(npc.encode()).hexdigest()

    def test_all_inference_uses_backend_and_output_is_resampled(self):
        r=self.request(instructions='A guard',seed=42,temperature=0,options={'future_option':7})
        self.assertEqual(r.status_code,200,r.text)
        audio,rate=sf.read(io.BytesIO(r.content));self.assertEqual((len(audio),rate),(96000,24000))
        self.assertEqual(self.backend.calls[-1][1]['instruction'],'A guard')
        self.assertEqual(self.backend.calls[-1][0].options,{'future_option':7})
        self.assertEqual(self.request(model='tts-1').status_code,200)

    def test_create_reuse_restart_and_legacy_reference_ignore_feature_cache(self):
        first=self.request(npc_id='guard',instructions='A guard')
        self.assertEqual(first.status_code,200,first.text)
        self.assertEqual(first.headers['x-voice-status'],'preparing')
        folder=self.folder();reference=(folder/'reference.wav').read_bytes()
        (folder/'prompt.npz').write_bytes(b'not a valid feature file')
        metadata=json.loads((folder/'metadata.json').read_text());metadata.update(version=2,clone_model='legacy-provider')
        (folder/'metadata.json').write_text(json.dumps(metadata))
        self.client.__exit__(None,None,None);self.start()
        self.assertEqual(self.request(npc_id='guard',instructions='Changed').status_code,200)
        self.assertEqual((folder/'reference.wav').read_bytes(),reference)
        self.assertEqual(self.backend.calls[-1][1]['mode'],'clone')
        self.assertEqual(sum(k.get('mode')=='design' for _,k in self.backend.calls),1)

    def test_short_first_line_is_extended_with_its_own_reference(self):
        r=self.request(npc_id='guard',instructions='A guard',input='short')
        self.assertEqual(r.status_code,200,r.text)
        meta=json.loads((self.folder()/'metadata.json').read_text())
        self.assertEqual(meta['status'],'ready');self.assertEqual(meta['reference_source'],'extended_dialogue')
        _,kwargs=self.backend.calls[-1]
        self.assertEqual(kwargs['mode'],'clone');self.assertIn('reference',kwargs);self.assertEqual(kwargs['reference_text'],'short')

    def test_cropped_and_possibly_truncated_extensions_do_not_guess_transcripts(self):
        from unittest.mock import patch
        self.assertEqual(self.request(npc_id='guard',instructions='A guard',input='short').status_code,200)
        engine=self.client.app.state.engine
        folder=self.folder()
        metadata=json.loads((folder/'metadata.json').read_text())
        for seconds,truncated in ((12,False),(2,True)):
            with self.subTest(seconds=seconds,truncated=truncated):
                sf.write(folder/'dialogue.wav',np.ones(24000*seconds)*.1,24000)
                (folder/'reference.wav').unlink(missing_ok=True)
                metadata.update(status='failed',may_be_truncated=truncated)
                engine.write_metadata(folder,metadata)
                with patch.object(engine.selector,'select',side_effect=[
                    ValueError('No complete reference'),(np.ones(44100*4)*.1,'Extended words','dialogue')]):
                    engine.prepare('guard')
                kwargs=self.backend.calls[-1][1]
                self.assertIsNone(kwargs['reference_text'])
                self.assertLessEqual(len(kwargs['reference'][0]),24000*10)

    def test_creation_is_serialized_and_recovery_preserves_first_recording(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(lambda _:self.request(npc_id='guard',instructions='A guard'),range(3)))
        self.assertTrue(all(r.status_code==200 for r in results))
        self.assertEqual(sum(k.get('mode')=='design' for _,k in self.backend.calls),1)
        original=(self.folder()/'dialogue.wav').read_bytes()
        self.backend.fail=True
        self.assertEqual(self.request(npc_id='guard').status_code,502)
        self.backend.fail=False
        self.assertEqual(self.request(npc_id='guard').status_code,200)
        self.assertEqual((self.folder()/'dialogue.wav').read_bytes(),original)

    def test_model_auth_bridge_and_backend_option_validation(self):
        self.assertEqual(self.client.get('/v1/models').status_code,401)
        self.assertEqual(self.client.get('/health').json()['backend'],'audio.cpp')
        self.assertEqual(self.request(model='missing').status_code,400)
        self.assertEqual(self.request(model='reader',npc_id='guard').status_code,400)
        self.assertEqual(self.request(npc_id='guard').status_code,400)
        fields={'npc_id':'sn_guard','instructions':'A guard','max_new_tokens':128}
        voice='sillynpc-v1:'+base64.urlsafe_b64encode(json.dumps(fields).encode()).decode().rstrip('=')
        self.assertEqual(self.request(voice=voice).status_code,200)
        self.assertEqual(self.request(model='reader',voice=voice).status_code,400)

    def test_streaming_real_chunks_and_context_fallback(self):
        r=self.request(model='reader',stream=True,response_format='pcm')
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(len(r.content),14400);self.assertTrue(self.backend.stream_closed)
        self.assertEqual(r.headers['x-audio-sample-rate'],'24000')
        self.assertEqual(self.request(stream=True,response_format='pcm').status_code,400)
        r=self.request(model='reader',context_before='Before. ',context_after=' After.')
        self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.headers['x-narration-context'],'fallback')
        self.assertEqual(self.backend.calls[-2][1]['text'],'Before. Hello After.')

    def test_first_audio_is_delivered_before_reference_preparation(self):
        original=self.backend.generate
        body_sent=[]
        def generate(profile,body,**kwargs):
            if kwargs.get('mode')=='clone': self.assertTrue(body_sent)
            return original(profile,body,**kwargs)
        self.backend.generate=generate
        async def exercise():
            app=create_app(config=CONFIG,backend=self.backend,voices_dir=self.storage.name)
            async with app.router.lifespan_context(app):
                payload=json.dumps({'model':'actor','input':'short','npc_id':'first','instructions':'A guard','response_format':'wav'}).encode()
                async def receive():return {'type':'http.request','body':payload,'more_body':False}
                async def send(message):
                    if message['type']=='http.response.body':
                        self.assertTrue(message['body'].startswith(b'RIFF'));body_sent.append(True)
                await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST','scheme':'http','path':'/v1/audio/speech','raw_path':b'/v1/audio/speech','query_string':b'','headers':[(b'content-type',b'application/json')],'server':('test',80),'client':('test',1)},receive,send)
        asyncio.run(exercise())

    def test_stream_closes_on_asgi_disconnect_before_next_inference(self):
        from starlette.requests import ClientDisconnect
        sent=[]
        def stream(profile,body):
            try:
                yield np.ones(2400,dtype=np.float32)*0.1
                self.fail('Client disconnected; second synthesis must not run')
            finally:
                self.backend.stream_closed=True
        self.backend.stream=stream
        async def exercise():
            app=create_app(config=CONFIG,backend=self.backend,voices_dir=self.storage.name)
            async with app.router.lifespan_context(app):
                payload=json.dumps({'model':'reader','input':'Hello','response_format':'pcm','stream':True}).encode()
                async def receive():return {'type':'http.request','body':payload,'more_body':False}
                async def send(message):
                    if message['type']=='http.response.body' and message['body']:
                        sent.append(message['body']);raise OSError('Disconnected')
                scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'POST','scheme':'http','path':'/v1/audio/speech','raw_path':b'/v1/audio/speech','query_string':b'','headers':[(b'content-type',b'application/json')],'server':('test',80),'client':('test',1)}
                with self.assertRaises(ClientDisconnect):await app(scope,receive,send)
        asyncio.run(exercise());self.assertEqual(len(sent),1);self.assertTrue(self.backend.stream_closed)

    def test_failed_background_extension_and_restart_retry_preserve_identity(self):
        original=self.backend.generate
        def generate(profile,body,**kwargs):
            if kwargs.get('mode')=='clone':raise RuntimeError('reference extension failed')
            return original(profile,body,**kwargs)
        self.backend.generate=generate
        r=self.request(npc_id='guard',instructions='A guard',input='short')
        self.assertEqual(r.status_code,200)
        folder=self.folder();audio=(folder/'dialogue.wav').read_bytes()
        self.assertEqual(json.loads((folder/'metadata.json').read_text())['status'],'failed')
        self.backend.generate=original
        self.client.__exit__(None,None,None);self.start()
        self.assertEqual(self.request(npc_id='guard').status_code,200)
        self.assertEqual((folder/'dialogue.wav').read_bytes(),audio)
        self.assertEqual(json.loads((folder/'metadata.json').read_text())['status'],'ready')

    def test_pending_record_resumes_after_restart(self):
        self.assertEqual(self.request(npc_id='guard',instructions='A guard').status_code,200)
        folder=self.folder();meta=json.loads((folder/'metadata.json').read_text());meta['status']='preparing'
        (folder/'metadata.json').write_text(json.dumps(meta));(folder/'reference.wav').unlink()
        self.client.__exit__(None,None,None);self.start()
        self.assertEqual(self.request(npc_id='guard').status_code,200)
        self.assertEqual(json.loads((folder/'metadata.json').read_text())['status'],'ready')
