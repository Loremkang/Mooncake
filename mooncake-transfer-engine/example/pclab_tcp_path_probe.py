#!/usr/bin/env python3
"""Bounded CPU/GPU TCP path probe. Correctness checked after every transfer.

No SGLang, accelerator profiler, persistent network changes, or model files.
The sender API time and receiver validation time are separate measurements.
"""
import argparse
import ctypes
import http.server
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def thread_counters():
    result = {}
    for p in Path('/proc/self/task').iterdir():
        try:
            v = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            result[p.name] = {'ticks': int(v[11]) + int(v[12]),
                              'comm': (p / 'comm').read_text().strip(),
                              'schedstat': [int(x) for x in (p/'schedstat').read_text().split()]}
        except FileNotFoundError:
            pass
    return result


def tcp_counters():
    result = {}
    for name in ('snmp', 'netstat'):
        lines = Path('/proc/net', name).read_text().splitlines()
        for header, values in zip(lines[::2], lines[1::2]):
            for k, v in zip(header.split()[1:], values.split()[1:]):
                if any(s in k for s in ('Retrans','Timeout','Reorder','DSACK','SACK',
                                       'Loss','Backlog','Listen','WinProbe','Segs','RcvQDrop')):
                    result[header.split()[0]+k] = int(v)
    return result


def initialize(ip):
    import mooncake.engine
    e = mooncake.engine.TransferEngine()
    assert e.initialize(ip, 'P2PHANDSHAKE', 'tcp', '') == 0
    return e, f'{ip}:{e.get_rpc_port()}', mooncake.engine.__file__


def receiver(args):
    import numpy as np
    size = args.blocks * args.block_mib * 1024 * 1024
    if args.memory == 'cuda':
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(args.gpu)
        buf = torch.zeros(size, dtype=torch.uint8, device=f'cuda:{args.gpu}')
        torch.cuda.synchronize()
        ptr = buf.data_ptr()
    else:
        buf = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(buf)
    engine, session, library = initialize(args.local_ip)
    assert engine.register_memory(ptr, size) == 0
    meta = {'pid': os.getpid(), 'session': session, 'ptr': ptr, 'bytes': size,
            'blocks': args.blocks, 'block_mib': args.block_mib,
            'memory': args.memory, 'gpu': args.gpu, 'library': library,
            'chunk_size': os.environ.get('MC_TCP_SLICE_SIZE', '65536'),
            'pool': os.environ.get('MC_TCP_ENABLE_CONNECTION_POOL', '0'),
            'strict_verify': args.strict_verify}
    prepared = {}
    rows = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            data = json.dumps(meta).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(data)

        def do_POST(self):
            d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            try:
                if self.path == '/prepare':
                    assert not prepared, 'previous iteration not verified'
                    pattern = int(d['pattern']); assert 1 <= pattern <= 254
                    if args.memory == 'cuda':
                        buf.zero_(); torch.cuda.synchronize()
                    else:
                        ctypes.memset(ptr, 0, size)
                    prepared.update(pattern=pattern, iteration=d['iteration'],
                                    counters=thread_counters(), real_ns=time.time_ns(),
                                    mono_ns=time.monotonic_ns())
                    result = {'ready': True}
                elif self.path == '/verify':
                    assert prepared and d['iteration'] == prepared['iteration']
                    # Snapshot TE CPU before validation work; counters still include
                    # HTTP control handling and are not pure TE-only CPU attribution.
                    before_verify = thread_counters()
                    start = time.perf_counter(); attempts = 0
                    while True:
                        attempts += 1
                        if args.memory == 'cuda':
                            arr = buf.cpu().numpy()
                        else:
                            arr = np.ctypeslib.as_array(buf).view(np.uint8)
                        ok = bool(np.all(arr == prepared['pattern']))
                        if ok or args.strict_verify or time.perf_counter() - start > 10:
                            break
                        time.sleep(.005)
                    result = {'ok': ok, 'iteration': prepared['iteration'],
                              'receiver_prepare_real_ns': prepared['real_ns'],
                              'receiver_verify_real_ns': time.time_ns(),
                              'verify_ms': (time.perf_counter()-start)*1000,
                              'verify_attempts': attempts,
                              'receiver_thread_schedstat_delta': {
                                  k: [x-y for x,y in zip(v['schedstat'],
                                      prepared['counters'].get(k,v)['schedstat'])]
                                  for k,v in before_verify.items()},
                              'receiver_threads_cpu_ms_before_verify': {
                                  k: {'comm': v['comm'], 'cpu_ms':
                                      (v['ticks']-prepared['counters'].get(k,{'ticks':v['ticks']})['ticks'])
                                      / os.sysconf('SC_CLK_TCK')*1000}
                                  for k,v in before_verify.items()}}
                    rows.append(result); save(args.root/'receiver-rows.json', rows)
                    prepared.clear()
                elif self.path == '/stop':
                    result = {'stopped': True}
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    raise ValueError('unknown endpoint')
                data = json.dumps(result).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(data)
            except Exception as exc:
                save(args.root/'receiver-failure.json', {'error': repr(exc)})
                self.send_error(500, repr(exc))

    server = http.server.HTTPServer((args.local_ip, args.http_port), Handler)
    server.timeout = 1
    save(args.root/'receiver-meta.json', meta)
    print(json.dumps(meta), flush=True)
    server.serve_forever(poll_interval=.05)
    server.server_close()
    engine.unregister_memory(ptr)


def sender(args):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, data=None):
        r = urllib.request.Request(args.server_url+path,
              data=json.dumps(data).encode() if data is not None else None,
              headers={'Content-Type': 'application/json'})
        with opener.open(r, timeout=15) as f:
            return json.load(f)
    meta = request('/')
    size = args.blocks * args.block_mib * 1024 * 1024
    assert meta['bytes'] == size and meta['blocks'] == args.blocks
    assert not args.strict_verify or meta.get('strict_verify'), 'receiver must reject first mismatch'
    if args.strict_verify:
        assert os.environ.get('MC_TCP_WRITE_REMOTE_FENCE') == '1', 'correctness fence required'
    buf = ctypes.create_string_buffer(size)
    ptr = ctypes.addressof(buf)
    engine, session, library = initialize(args.local_ip)
    assert engine.register_memory(ptr, size) == 0
    block_size = args.block_mib * 1024 * 1024
    source = [ptr + i*block_size for i in range(args.blocks)]
    target = [meta['ptr'] + i*block_size for i in range(args.blocks)]
    rows = []
    result = {'sender_library': library, 'receiver': meta,
              'remote_fence': os.environ.get('MC_TCP_WRITE_REMOTE_FENCE', '0'),
              'sender_chunk': os.environ.get('MC_TCP_SLICE_SIZE', '65536'),
              'pool': os.environ.get('MC_TCP_ENABLE_CONNECTION_POOL', '0'),
              'strict_verify': args.strict_verify, 'idle_ms': args.idle_ms,
              'bytes_per_iteration': size, 'rows': rows, 'complete': False}
    save(args.root/'result.json', result)
    save(args.root/'sender-meta.json', {'pid':os.getpid(), 'session':session,
                                      'real_ns':time.time_ns()})
    try:
        for i in range(args.warmup + args.iterations):
            pattern = i % 250 + 1
            ctypes.memset(ptr, pattern, size)
            request('/prepare', {'pattern': pattern, 'iteration': i})
            tcp_before = tcp_counters()
            real_start = time.time_ns()
            start = time.perf_counter()
            rc = engine.batch_transfer_sync_write(meta['session'], source, target,
                                                  [block_size]*args.blocks)
            api_ms = (time.perf_counter()-start)*1000
            real_end = time.time_ns()
            tcp_after = tcp_counters()
            if rc != 0:
                raise RuntimeError(f'TE transfer error {rc}')
            check = request('/verify', {'iteration': i})
            confirmation_ms = (time.perf_counter()-start)*1000
            row = {'iteration':i, 'warmup':i<args.warmup, 'api_ms':api_ms,
                   'sender_start_real_ns':real_start,'sender_end_real_ns':real_end,
                   'sender_tcp_delta':{k:tcp_after[k]-v for k,v in tcp_before.items()
                                       if tcp_after[k]!=v},
                   'confirmation_ms':confirmation_ms, **check}
            rows.append(row); save(args.root/'result.json', result)
            print(json.dumps(row), flush=True)
            if not check['ok'] or (args.strict_verify and check['verify_attempts'] != 1):
                raise RuntimeError('first full-buffer verification failed')
            if i + 1 < args.warmup + args.iterations:
                time.sleep(args.idle_ms / 1000)
        measured = [r for r in rows if not r['warmup']]
        result['api_ms_median'] = statistics.median(r['api_ms'] for r in measured)
        result['api_ms_mean'] = statistics.mean(r['api_ms'] for r in measured)
        result['api_effective_gbps'] = size*8/result['api_ms_mean']/1e6
        result['complete'] = True
    except Exception as exc:
        result['failure'] = repr(exc)
        raise
    finally:
        save(args.root/'result.json', result)
        # Only stops this isolated receiver, including on failure. It has no
        # dependency on the model service. Preserve all result/log files.
        try:
            request('/stop', {})
        except Exception:
            pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['receiver','sender'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--local-ip', required=True)
    p.add_argument('--server-url', default='http://192.168.89.8:35867')
    p.add_argument('--http-port', type=int, default=35867)
    p.add_argument('--memory', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--blocks', type=int, default=30)
    p.add_argument('--block-mib', type=int, default=9)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--strict-verify', action='store_true')
    p.add_argument('--idle-ms', type=int, default=0)
    a = p.parse_args()
    assert 0 <= a.idle_ms <= 10000
    assert 0 < a.blocks*a.block_mib <= 600
    a.root.mkdir(parents=True, exist_ok=True)
    assert not (a.root/'result.json').exists(), 'preserve existing attempt'
    (receiver if a.role == 'receiver' else sender)(a)
