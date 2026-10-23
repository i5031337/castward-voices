import base64
import io
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import httpx
import numpy as np
import soundfile as sf
from audiocpp_backend import AudioCppBackend
from backend_config import ServerConfig
from server import SpeechRequest


def settings():
    return ServerConfig.model_validate({"default_model":"actor", "models":{
        "actor":{"backend_model":"future-model", "design_model":"designer", "clone_model":"cloner",
                 "instruction_option":"caption", "language_map":{"English":"en"},
                 "design_options":{"strategy":"design"}},
        "reader":{"backend_model":"reader", "streaming_model":"reader-stream", "streaming_sample_rate":44100,
                  "native_speed":True}}})


class BackendTests(unittest.TestCase):
    def test_routed_generation_streaming_discovery_voices_and_auth(self):
        config = settings().model_copy(update={
            'backend_url': 'http://gpu.local',
            'backend_routes': {'reader': 'http://cpu.local/', 'reader-stream': 'http://cpu.local/'},
            'backend_api_key_env': 'TEST_NATIVE_KEY',
        })
        calls = []
        wav = io.BytesIO()
        sf.write(wav, np.ones(2400)*0.1, 24000, format='WAV')
        pcm = base64.b64encode(np.ones(4410, dtype='<i2').tobytes()).decode()
        def handle(request):
            calls.append((request.url.host, request.url.path))
            self.assertEqual(request.headers['Authorization'], 'Bearer native-key')
            if request.url.path == '/v1/models':
                # A stale GPU server must not shadow metadata owned by the CPU route.
                return httpx.Response(200, json={'data': [
                    {'id': name, 'source': request.url.host}
                    for name in ('future-model', 'reader', 'reader-stream')]})
            if request.url.path == '/v1/audio/voices':
                self.assertEqual(request.url.host, 'cpu.local')
                return httpx.Response(200, json={'voices': ['heart']})
            body = json.loads(request.content)
            expected = 'cpu.local' if body['model'].startswith('reader') else 'gpu.local'
            self.assertEqual(request.url.host, expected)
            if body.get('stream_format'):
                return httpx.Response(200, text='data: '+json.dumps({'type':'speech.audio.delta','audio':pcm})+'\n\ndata: [DONE]\n\n')
            return httpx.Response(200, content=wav.getvalue())
        with patch.dict('os.environ', {'TEST_NATIVE_KEY': 'native-key'}):
            backend = AudioCppBackend(config, transport=httpx.MockTransport(handle))
        try:
            entries = {m['id']: m['source'] for m in backend.discover()}
            self.assertEqual(entries, {'future-model':'gpu.local','reader':'cpu.local','reader-stream':'cpu.local'})
            self.assertEqual(backend.voices(config.models['reader']), ['heart'])
            for name in ('actor','reader'):
                audio, rate = backend.generate(config.models[name], SpeechRequest(model=name,input='Hello'))
                self.assertEqual((len(audio),rate), (2400,24000))
            chunks = list(backend.stream(config.models['reader'], SpeechRequest(model='reader',input='Hello')))
            self.assertEqual(len(chunks[0]),2400)
        finally:
            backend.close()
        self.assertTrue(all(client.is_closed for client in backend._clients.values()))
        self.assertEqual(calls.count(('cpu.local','/v1/models')),1)

    def test_routed_discovery_failure_is_not_hidden(self):
        config = settings().model_copy(update={'backend_routes': {'reader':'http://cpu.local'}})
        backend = AudioCppBackend(config, transport=httpx.MockTransport(
            lambda req: httpx.Response(503,json={'error':'CPU unavailable'}) if req.url.host=='cpu.local'
            else httpx.Response(200,json={'data':[]})))
        try:
            with self.assertRaisesRegex(RuntimeError,'CPU unavailable'):
                backend.discover()
        finally:
            backend.close()

    def test_model_mapping_defaults_reference_and_full_width_seed(self):
        backend = AudioCppBackend(settings())
        body = SpeechRequest(model="actor",input="Hello",seed=2**63+9,options={"guidance":2})
        request = backend.payload(settings().models['actor'],body,mode="design",instruction="A voice")
        self.assertEqual(request['model'],'designer')
        self.assertEqual(request['seed'],str(2**63+9))
        self.assertEqual(request['options'],{'strategy':'design','language':'en','caption':'A voice','guidance':2})
        self.assertNotIn('temperature',request['options'])
        request = backend.payload(settings().models['actor'],body,mode="clone",reference=(np.ones(2400)*0.1,24000),reference_text="Reference")
        audio,rate = sf.read(io.BytesIO(base64.b64decode(request['voice_ref']['data'])))
        self.assertEqual((len(audio),rate),(2400,24000))
        self.assertEqual(request['options']['reference_text'],'Reference')
        backend.close()

    def test_generation_checks_status_and_rejects_invalid_waveform(self):
        def handle(request):
            return httpx.Response(400,json={'error':{'message':'Unsupported language'}})
        backend=AudioCppBackend(settings(),transport=httpx.MockTransport(handle))
        with self.assertRaisesRegex(ValueError,'Unsupported language'):
            backend.generate(settings().models['actor'],SpeechRequest(model='actor',input='Hello'))
        backend.close()

    def test_stream_decodes_sse_and_resamples_complete_audio_events(self):
        audio=np.full(4410,1000,dtype='<i2')
        payload=base64.b64encode(audio.tobytes()).decode()
        data='data: '+json.dumps({'type':'speech.audio.delta','audio':payload})+'\n\ndata: [DONE]\n\n'
        backend=AudioCppBackend(settings(),transport=httpx.MockTransport(lambda req:httpx.Response(200,text=data)))
        chunks=list(backend.stream(settings().models['reader'],SpeechRequest(model='reader',input='Hello',response_format='pcm')))
        self.assertEqual(len(chunks),1)
        self.assertEqual(len(chunks[0]),2400)
        backend.close()

    def test_stream_errors_and_truncation_are_not_reported_as_success(self):
        for data in ('data: '+json.dumps({'type':'error','error':'broken'})+'\n\n','data: '+json.dumps({'type':'speech.audio.delta','audio':base64.b64encode(b'x').decode()})+'\n\n',''):
            backend=AudioCppBackend(settings(),transport=httpx.MockTransport(lambda req:httpx.Response(200,text=data)))
            with self.assertRaises(RuntimeError):
                list(backend.stream(settings().models['reader'],SpeechRequest(model='reader',input='Hello')))
            backend.close()

    def test_stereo_stream_is_downmixed_before_resampling(self):
        profile = settings().models['reader'].model_copy(update={'streaming_channels':2})
        audio = np.tile(np.array([2000,0],dtype='<i2'),(4410,1))
        data = 'data: '+json.dumps({'type':'speech.audio.delta','audio':base64.b64encode(audio.tobytes()).decode()})+'\n\ndata: [DONE]\n\n'
        backend = AudioCppBackend(settings(),transport=httpx.MockTransport(lambda req:httpx.Response(200,text=data)))
        chunks = list(backend.stream(profile,SpeechRequest(model='reader',input='Hello')))
        self.assertEqual(chunks[0].shape,(2400,))
        self.assertAlmostEqual(float(chunks[0][100]),1000/32768,places=4)
        backend.close()

    def test_profile_validation_rejects_unknown_alias_and_unspecified_stream_rate(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ServerConfig.model_validate({'default_model':'x','models':{'x':{'backend_model':'x','streaming_model':'stream'}}})
        with self.assertRaises(ValidationError):
            ServerConfig.model_validate({'default_model':'x','models':{'x':{'backend_model':'x'}},'aliases':{'bad':'missing'}})
