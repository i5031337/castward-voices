"""Live FireRed design/clone/NPC check against an already running audio.cpp service.

Run from the repository root: .venv/bin/python tests/smoke_firered.py --backend-url http://127.0.0.1:7862
Artifacts and timings go to out/bench/firered-smoke/; NPC storage is temporary.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend_config import load_config
from server import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', default='http://127.0.0.1:7862')
    parser.add_argument('--output', type=Path, default=Path('out/bench/firered-smoke'))
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parents[1] / 'backend.json')
    config.backend_url = args.backend_url
    config.backend_routes = {}
    config.backend_timeout = 600
    config.models = {'fireredtts3': config.models['fireredtts3']}
    config.default_model = 'fireredtts3'
    config.aliases = {}
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, raw, elapsed):
        audio, rate = sf.read(io.BytesIO(raw))
        if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < .001:
            raise RuntimeError(f'{name}: empty, nonfinite, or silent output')
        (args.output / f'{name}.wav').write_bytes(raw)
        item = {'name': name, 'elapsed_seconds': elapsed, 'audio_seconds': len(audio) / rate, 'sample_rate': rate}
        results.append(item)
        print(json.dumps(item), flush=True)

    def speech(text, **kwargs):
        return {'model': 'fireredtts3', 'input': text, 'response_format': 'wav', 'seed': 1895,
                'options': {'num_inference_steps': 4}, **kwargs}

    with tempfile.TemporaryDirectory() as voices, TestClient(create_app(config=config, voices_dir=voices)) as client:
        metadata = client.get('/ui/config')
        metadata.raise_for_status()
        started = time.perf_counter()
        response = client.post('/ui/generate', json={'mode': 'design', 'speech': speech(
            'The road ahead is long, but there is always a place for you by the fire.',
            instructions='A warm male storyteller with a clear, natural delivery.')})
        response.raise_for_status()
        take = response.json()
        record('design', base64.b64decode(take['audio']), time.perf_counter() - started)
        started = time.perf_counter()
        response = client.post('/ui/generate', json={'mode': 'clone', 'reference_audio': take['reference_audio'],
            'reference_text': take['transcript'], 'speech': speech('The gate is open. Welcome back, traveler.')})
        response.raise_for_status()
        record('clone', base64.b64decode(response.json()['audio']), time.perf_counter() - started)
        for name, body in (
            ('preset', speech('Welcome home, traveler.', voice='echo')),
            ('npc-first', speech('Hello.', npc_id='firered-smoke', instructions='A calm older guard with a deep voice.')),
            ('npc-reuse', speech('The watch is over. You may pass through the gate.', npc_id='firered-smoke')),
        ):
            started = time.perf_counter()
            response = client.post('/v1/audio/speech', json=body)
            response.raise_for_status()
            record(name, response.content, time.perf_counter() - started)
            if name == 'npc-first':
                status = client.get('/v1/voices/firered-smoke').json()
                if status['status'] != 'ready':
                    raise RuntimeError(f'NPC preparation failed: {status}')
                results[-1]['reference_seconds'] = status['reference_seconds']
                saved = json.loads(next(Path(voices).glob('*/metadata.json')).read_text())
                results[-1]['reference_source'] = saved['reference_source']
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
