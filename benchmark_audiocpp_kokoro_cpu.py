"""Benchmark a separate audio.cpp CPU Kokoro server configured with four threads."""
import argparse
import base64
import io
import json
from pathlib import Path
import statistics
import time

import httpx
import numpy as np
import soundfile as sf

from benchmark_audiocpp_qwen import TEXTS


def offline(client, text):
    started = time.perf_counter()
    response = client.post('/v1/audio/speech', json={
        'model': 'kokoro', 'input': text, 'voice': 'af_heart',
        'response_format': 'wav', 'seed': '1895',
    })
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    audio, rate = sf.read(io.BytesIO(response.content))
    if len(audio) < rate or not np.isfinite(audio).all():
        raise RuntimeError('Invalid CPU Kokoro waveform')
    native = float(response.headers['X-AudioCPP-Wall-Ms']) / 1000
    duration = len(audio) / rate
    return {'client_seconds': elapsed, 'native_seconds': native,
            'audio_seconds': duration, 'rtf': native / duration}


def streaming(client, budget):
    started = time.perf_counter()
    first = None
    total = 0
    completed = False
    payload = {
        'model': 'kokoro-stream', 'input': TEXTS['long'], 'voice': 'af_heart',
        'response_format': 'pcm', 'stream_format': 'sse', 'seed': '1895',
        'options': {'text_chunk_size': budget} if budget else {},
    }
    with client.stream('POST', '/v1/audio/speech', json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                completed = True
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(event)
            if event.get('type') == 'speech.audio.delta':
                chunk = base64.b64decode(event['audio'], validate=True)
                total += len(chunk)
                if chunk and first is None:
                    first = time.perf_counter() - started
    if not completed or first is None or total <= 48000 or total % 2:
        raise RuntimeError('Incomplete CPU Kokoro PCM stream')
    return {'first_pcm_seconds': first, 'client_seconds': time.perf_counter() - started,
            'audio_seconds': total / 48000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:17863')
    parser.add_argument('--output', type=Path, default=Path('out/bench/audiocpp_kokoro_cpu.json'))
    args = parser.parse_args()
    rows = []
    result = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'threads': 4, 'backend': 'cpu', 'voice': 'af_heart', 'seed': 1895,
        'texts': TEXTS, 'records': rows,
        'notes': [
            'Use the same Q8 Kokoro file as CUDA; isolated CPU server, four threads, max_loaded_models=1.',
            'Each workload uses one warmup and three measured requests; streaming consumes the whole passage.',
            'First PCM is the first decoded native SSE audio payload at the client.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        rows.append(row)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    with httpx.Client(base_url=args.url, timeout=240) as client:
        health = client.get('/health')
        health.raise_for_status()
        if health.json().get('backend') != 'cpu':
            raise RuntimeError('Use an isolated native server with backend=cpu')
        for length, text in TEXTS.items():
            for trial in range(4):
                record({'length': length, 'trial': trial, 'warmup': trial == 0,
                        **offline(client, text)})
        for budget in (None, 80):
            for trial in range(4):
                record({'length': 'long', 'stream': True, 'text_chunk_size': budget or 240,
                        'trial': trial, 'warmup': trial == 0, **streaming(client, budget)})

    result['summary'] = []
    for length in TEXTS:
        measured = [r for r in rows if r['length'] == length and not r['warmup'] and not r.get('stream')]
        result['summary'].append({'length': length, **{
            key: statistics.median(r[key] for r in measured)
            for key in ('native_seconds', 'client_seconds', 'audio_seconds', 'rtf')
        }})
    for budget, key in ((240, 'median_first_pcm_seconds'), (80, 'median_first_pcm_80_seconds')):
        result[key] = statistics.median(
            r['first_pcm_seconds'] for r in rows
            if r.get('text_chunk_size') == budget and not r['warmup']
        )
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
