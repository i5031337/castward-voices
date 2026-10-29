import json
from pathlib import Path
import socket
import tempfile
import unittest

from launch_config import check_ports, prepare_configs, prepare_cpu_configs
from backend_config import load_config
from audiocpp_backend import AudioCppBackend
from server import SpeechRequest
import numpy as np


class LaunchConfigTests(unittest.TestCase):
    def test_all_cpu_workflow_designs_then_clones_without_designer_options(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native, speech = root/'native.json', root/'speech.json'
            source = {'backend':'cuda', 'device':1, 'models':[
                {'id':'ming','family':'ming3_tts','path':'ming.gguf','device':1},
                {'id':'pocket-english','family':'pocket_tts','path':'pocket.gguf','device':1},
            ]}
            native.write_text(json.dumps(source))
            speech.write_bytes((repository/'backend.json').read_bytes())
            before = native.read_bytes(), speech.read_bytes()
            prepare_cpu_configs(native,speech,root/'runtime',cpu_port=12342)
            cpu = json.loads((root/'runtime/cpu.json').read_text())
            config = load_config(root/'runtime/backend.json')
            self.assertEqual((cpu['backend'],cpu['threads'],cpu['max_loaded_models']),('cpu',8,2))
            self.assertNotIn('device',cpu)
            self.assertTrue(all('device' not in m and m['lazy'] for m in cpu['models']))
            self.assertEqual(cpu['models'][0]['path'],str(root/'ming.gguf'))
            self.assertEqual(config.backend_url,'http://127.0.0.1:12342')
            self.assertEqual(config.models['narrator'].backend_model,'pocket-english')
            self.assertIsNone(config.models['narrator'].streaming_model)
            self.assertEqual(config.models['qwen3-tts'].clone_model,'qwen-base-0.6b')
            profile = config.models['npc']
            backend = AudioCppBackend(config)
            try:
                body = SpeechRequest(model='npc',input='Welcome back.')
                design = backend.payload(profile,body,mode='design',instruction='A warm voice')
                self.assertEqual(design['model'],'ming')
                self.assertEqual(design['options']['template_name'],'voice_design')
                clone = backend.payload(profile,body,mode='clone',reference=(np.ones(24000)*.1,24000),reference_text='Welcome')
                self.assertEqual(clone['model'],'pocket-english')
                self.assertEqual(clone['options'],{})
                preset = backend.payload(profile,body)
                self.assertEqual((preset['model'],preset['voice']),('pocket-english','alba'))
            finally:
                backend.close()
            self.assertEqual(before,(native.read_bytes(),speech.read_bytes()))
            source['models'].pop()
            native.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError,'PocketTTS'):
                prepare_cpu_configs(native,speech,root/'runtime')

    def test_cpu_mode_splits_models_limits_gpu_residency_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / 'native.json'
            speech = root / 'speech.json'
            source = {'backend':'cuda','max_loaded_models':2,'models':[
                {'id':'actor','family':'future_tts','path':'models/actor.gguf','device':1},
                {'id':'reader','family':'kokoro_tts','path':'models/kokoro.gguf','device':1},
                {'id':'stream','family':'kokoro_tts','path':'models/kokoro.gguf','device':1},
            ]}
            settings = {'default_model':'actor','models':{
                'actor':{'backend_model':'actor'},
                'reader':{'backend_model':'reader','streaming_model':'stream','streaming_sample_rate':24000},
            },'alignment_model':'alignment.pt'}
            native.write_text(json.dumps(source))
            speech.write_text(json.dumps(settings))
            before = native.read_bytes(), speech.read_bytes()
            prepare_configs(native,speech,root/'runtime',gpu_device=2,gpu_port=12341,cpu_port=12342)
            gpu,cpu,api = [json.loads((root/'runtime'/name).read_text()) for name in ('gpu.json','cpu.json','backend.json')]
            self.assertEqual((gpu['backend'],gpu['max_loaded_models'],gpu['device']),('cuda',1,2))
            self.assertEqual([m['id'] for m in gpu['models']],['actor'])
            self.assertEqual(gpu['models'][0]['device'],2)
            self.assertEqual(gpu['models'][0]['path'],str(root/'models/actor.gguf'))
            self.assertEqual(cpu['backend'],'cpu')
            self.assertEqual({m['id'] for m in cpu['models']},{'reader','stream'})
            self.assertTrue(all('device' not in m for m in cpu['models']))
            self.assertEqual(api['backend_routes'],{'reader':'http://127.0.0.1:12342','stream':'http://127.0.0.1:12342'})
            self.assertEqual(api['backend_url'],'http://127.0.0.1:12341')
            self.assertEqual(api['alignment_model'],str(root/'alignment.pt'))
            self.assertEqual(api['models']['reader']['streaming_options']['text_chunk_size'],80)
            self.assertEqual(before,(native.read_bytes(),speech.read_bytes()))
            settings['models']['reader']['streaming_options']={'text_chunk_size':120}
            speech.write_text(json.dumps(settings))
            prepare_configs(native,speech,root/'runtime')
            api=json.loads((root/'runtime/backend.json').read_text())
            self.assertEqual(api['models']['reader']['streaming_options']['text_chunk_size'],120)

    def test_rejects_occupied_native_ports(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            with self.assertRaisesRegex(ValueError,'occupied'):
                check_ports(sock.getsockname()[1])
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1',0))
            sock.listen()
            with self.assertRaisesRegex(ValueError,'occupied'):
                check_ports(sock.getsockname()[1])
