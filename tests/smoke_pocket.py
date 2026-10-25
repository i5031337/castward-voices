"""Live PocketTTS preset, reference cloning, and saved-NPC reuse on CPU.

Start audiocpp-server.pocket.json, then run .venv/bin/python tests/smoke_pocket.py.
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
from server import create_app, SpeechRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', default='http://127.0.0.1:7865')
    parser.add_argument('--output', type=Path, default=Path('out/bench/pocket-smoke'))
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parents[1] / 'backend-pocket.json')
    config.backend_url = args.backend_url
    config.backend_timeout = 600
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, raw, elapsed):
        audio, rate = sf.read(io.BytesIO(raw), dtype='float32')
        if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < .001:
            raise RuntimeError(f'{name}: empty, nonfinite, or silent audio')
        (args.output / f'{name}.wav').write_bytes(raw)
        item = {'name': name, 'elapsed_seconds': elapsed, 'audio_seconds': len(audio) / rate,
                'sample_rate': rate, 'rtf': elapsed / (len(audio) / rate)}
        results.append(item)
        print(json.dumps(item), flush=True)
        return audio, rate

    def speech(text, **kwargs):
        return {'model': 'pocket-tts', 'input': text, 'seed': 1895, 'response_format': 'wav', **kwargs}

    with tempfile.TemporaryDirectory() as voices, TestClient(create_app(config=config, voices_dir=voices)) as client:
        response = client.get('/ui/config')
        response.raise_for_status()
        profile = response.json()['profiles'][0]
        if 'alba' not in profile['voices']:
            raise RuntimeError('Packaged alba voice missing from discovery')
        text = 'The road ahead is long, but there is always a place for you by the fire.'
        started = time.perf_counter()
        response = client.post('/v1/audio/speech', json=speech(text, voice='alba'))
        response.raise_for_status()
        audio, rate = record('preset', response.content, time.perf_counter() - started)
        reference = base64.b64encode(response.content).decode()
        started = time.perf_counter()
        response = client.post('/ui/generate', json={'mode': 'clone', 'reference_audio': reference,
            'reference_text': text, 'speech': speech('The gate is open. Welcome back, traveler.')})
        response.raise_for_status()
        record('clone', base64.b64decode(response.json()['audio']), time.perf_counter() - started)
        # Import the generated recording into temporary NPC storage only.
        engine = client.app.state.engine
        engine.save_dialogue(SpeechRequest(**speech(text, npc_id='pocket-smoke')), 'pocket-tts',
                             'Packaged alba recording', 1895, audio, rate)
        started = time.perf_counter()
        response = client.post('/v1/audio/speech', json=speech('You may pass through the gate.', npc_id='pocket-smoke'))
        response.raise_for_status()
        record('npc-reuse', response.content, time.perf_counter() - started)
        status = client.get('/v1/voices/pocket-smoke').json()
        if status['status'] != 'ready':
            raise RuntimeError(f'NPC preparation failed: {status}')
        results[-1]['reference_seconds'] = status['reference_seconds']
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
