import json
from pathlib import Path
import socket
import tempfile
import unittest

from launch_config import check_ports, prepare_configs


class LaunchConfigTests(unittest.TestCase):
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
