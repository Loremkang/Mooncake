#!/usr/bin/env python3
"""Coordinate six independent Mooncake senders using only the H20 clock.

Ready barrier starts after buffer preparation. Completion barrier ends after
all receivers' layer tails arrive, before any full-buffer verification. Thus
the shared interval excludes filling, validation, and between-round idle time.
It includes control HTTP latency and is not a packet one-way latency.
"""
import argparse
import http.server
import json
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=int, default=6)
    parser.add_argument('--nic', default='ens49f0np0')
    parser.add_argument('--port', type=int, default=35880)
    args = parser.parse_args()
    assert 1 <= args.sources <= 24
    args.root.mkdir(parents=True, exist_ok=False)
    condition = threading.Condition()
    rounds = {}
    stop = []

    def counters():
        p = Path('/sys/class/net') / args.nic
        keys = ['carrier_changes', 'carrier_down_count', 'carrier_up_count',
                'statistics/rx_bytes', 'statistics/tx_bytes', 'statistics/rx_dropped',
                'statistics/tx_dropped', 'statistics/rx_errors', 'statistics/tx_errors']
        return {k: int((p / k).read_text()) for k in keys}

    def persist():
        tmp = args.root / 'rounds.tmp'
        tmp.write_text(json.dumps({'stop': stop, 'rounds': rounds}, indent=2) + '\n')
        tmp.replace(args.root / 'rounds.json')

    def wait(predicate):
        deadline = time.monotonic() + 50
        while not predicate():
            if stop:
                raise RuntimeError(stop)
            left = deadline - time.monotonic()
            if left <= 0:
                stop.append('multi-source barrier timeout')
                condition.notify_all()
                persist()
                raise TimeoutError(stop)
            condition.wait(left)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with condition:
                self.reply({'sources': args.sources, 'rounds': len(rounds), 'stop': stop})

        def do_POST(self):
            try:
                d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with condition:
                    source = d['source']
                    assert 0 <= source < args.sources
                    if self.path == '/abort':
                        stop.append({'source': source, 'error': d.get('error')})
                        condition.notify_all()
                        persist()
                        self.reply({'aborted': True})
                        return
                    assert not stop, stop
                    r = rounds.setdefault(d['round'], {'ready': {}, 'completed': {}})
                    if self.path == '/ready':
                        assert source not in r['ready'], 'duplicate source ready'
                        assert d['bytes'] > 0
                        r['ready'][source] = {'at_ns': time.monotonic_ns(),
                                              'bytes': d['bytes'], 'warmup': d['warmup']}
                        if len(r['ready']) == args.sources:
                            assert len({x['warmup'] for x in r['ready'].values()}) == 1
                            r['nic_before'] = counters()
                            r['start_mono_ns'] = time.monotonic_ns()
                            condition.notify_all()
                        wait(lambda: 'start_mono_ns' in r)
                        self.reply({'start_mono_ns': r['start_mono_ns']})
                    elif self.path == '/complete':
                        assert source in r['ready'] and source not in r['completed']
                        r['completed'][source] = {'at_ns': time.monotonic_ns(),
                            'api_span_ms': d['api_span_ms'],
                            'completion_notice_ms': d['completion_notice_ms']}
                        if len(r['completed']) == args.sources:
                            r['end_mono_ns'] = time.monotonic_ns()
                            r['nic_after'] = counters()
                            assert all(r['nic_after'][k] == v for k, v in r['nic_before'].items()
                                       if not k.endswith(('/rx_bytes', '/tx_bytes'))), 'NIC carrier/drop/error changed'
                            span = (r['end_mono_ns'] - r['start_mono_ns']) / 1e9
                            payload = sum(x['bytes'] for x in r['ready'].values())
                            r['summary'] = {'round': d['round'],
                                'warmup': next(iter(r['ready'].values()))['warmup'],
                                'h20_span_ms': span * 1000, 'payload_bytes': payload,
                                'payload_gbps': payload * 8 / span / 1e9,
                                'nic_rx_gbps': (r['nic_after']['statistics/rx_bytes']
                                    - r['nic_before']['statistics/rx_bytes']) * 8 / span / 1e9}
                            persist()
                            condition.notify_all()
                        wait(lambda: 'summary' in r)
                        self.reply(r['summary'])
                    else:
                        raise ValueError('unknown coordinator endpoint')
            except Exception as exc:
                with condition:
                    if not stop:
                        stop.append(repr(exc))
                    condition.notify_all()
                    persist()
                self.send_error(500, repr(exc))

    server = http.server.ThreadingHTTPServer(('192.168.89.8', args.port), Handler)
    server.daemon_threads = True
    (args.root / 'ready.json').write_text(json.dumps({'pid': __import__('os').getpid(),
        'port': args.port, 'sources': args.sources}) + '\n')
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
