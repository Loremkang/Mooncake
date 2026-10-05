#!/usr/bin/env python3
"""Bounded, independently verified RAM -> RAM/CUDA Mooncake TCP diagnostic.

Uses one TransferEngine with 1 or 4 concurrent batch writers. Each writer has
its own registered source and destination region. The default batch is 30 x
9 MiB, matching one PP0 rank's 8K MLA KV payload, without SGLang or NPU pack.
Old TCP WRITE completion is local: receiver verification is timed separately.
No route, NIC, ARP, clock, model or persistent system settings are changed.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import http.server
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def threads():
    result = {}
    for p in Path('/proc/self/task').iterdir():
        try:
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            result[p.name] = {'ticks': int(stat[11]) + int(stat[12]),
                             'comm': (p / 'comm').read_text().strip(),
                             'sched': [int(x) for x in (p / 'schedstat').read_text().split()]}
        except FileNotFoundError:
            pass
    return result


def thread_delta(before, after):
    return {k: {'comm': v['comm'], 'cpu_ms': (v['ticks'] - before[k]['ticks'])
                * 1000 / os.sysconf('SC_CLK_TCK'),
                'runqueue_ms': (v['sched'][1] - before[k]['sched'][1]) / 1e6}
            for k, v in after.items() if k in before}


def tcp_counters():
    result = {}
    for name in ('snmp', 'netstat'):
        lines = Path('/proc/net', name).read_text().splitlines()
        for keys, vals in zip(lines[::2], lines[1::2]):
            for k, v in zip(keys.split()[1:], vals.split()[1:]):
                if any(s in k for s in ('Retrans', 'Timeout', 'Reorder', 'DSACK',
                                       'Loss', 'Backlog', 'Listen', 'Segs', 'ActiveOpens',
                                       'PassiveOpens', 'RcvQDrop')):
                    result[keys.split()[0] + k] = int(v)
    return result


def initialize(ip):
    import mooncake.engine
    engine = mooncake.engine.TransferEngine()
    assert engine.initialize(ip, 'P2PHANDSHAKE', 'tcp', '') == 0
    library = Path(mooncake.engine.__file__)
    return engine, {'pid': os.getpid(), 'session': f'{ip}:{engine.get_rpc_port()}',
                    'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                    'pool': os.environ.get('MC_TCP_ENABLE_CONNECTION_POOL', 'unset'),
                    'slice_size': os.environ.get('MC_TCP_SLICE_SIZE', 'unset'),
                    'remote_fence': os.environ.get('MC_TCP_WRITE_REMOTE_FENCE', 'unset')}


def receiver(args):
    import numpy as np
    size = args.blocks * args.block_mib * 1024 * 1024
    total = size * args.writers
    if args.memory == 'cuda':
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(args.gpu)
        free, capacity = torch.cuda.mem_get_info()
        assert free > total + 512 * 1024 * 1024, ('insufficient GPU headroom', free, total)
        buf = torch.zeros(total, dtype=torch.uint8, device=f'cuda:{args.gpu}')
        torch.cuda.synchronize()
        ptr = buf.data_ptr()
        # Read the tail of every layer with one pitched D2H call per writer.
        # TCP ServerSession copies each layer sequentially, so its tail arriving
        # implies its preceding bytes have been copied. Full verification still
        # follows, but no full-buffer D2H copy disturbs an incomplete transfer.
        runtime_paths = {l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines()
                         if '/libcudart.so' in l and not l.endswith('(deleted)')}
        assert len(runtime_paths) == 1, runtime_paths
        cudart = ctypes.CDLL(next(iter(runtime_paths)))
        cudart.cudaMemcpy2D.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
        cudart.cudaMemcpy2D.restype = ctypes.c_int
        tail_buf = ctypes.create_string_buffer(args.blocks * 8)
    else:
        buf = ctypes.create_string_buffer(total)
        ptr = ctypes.addressof(buf)
    engine, meta = initialize(args.local_ip)
    assert engine.register_memory(ptr, total) == 0
    meta.update(ptr=ptr, bytes_per_writer=size, writers=args.writers,
                blocks=args.blocks, memory=args.memory, gpu=args.gpu)
    active, rows = {}, []

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
            self.reply(meta)

        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/prepare':
                    assert not active, 'prior round not verified'
                    if args.memory == 'cuda':
                        buf.zero_()
                        torch.cuda.synchronize()
                    else:
                        ctypes.memset(ptr, 0, total)
                    active.update(round=data['round'], patterns=data['patterns'], before=threads())
                    assert len(active['patterns']) == args.writers
                    self.reply({'ready': True})
                elif self.path == '/await-complete':
                    assert active and active['round'] == data['round']
                    start = time.perf_counter_ns()
                    attempts = 0
                    block = args.block_mib * 1024 * 1024
                    while True:
                        attempts += 1
                        checks = []
                        for w, pattern in enumerate(active['patterns']):
                            if args.memory == 'cuda':
                                rc = cudart.cudaMemcpy2D(ctypes.addressof(tail_buf), 8,
                                    ptr + w * size + block - 8, block, 8, args.blocks, 2)
                                assert rc == 0, ('cudaMemcpy2D', rc)
                                checks.append(tail_buf.raw == bytes([pattern]) * (args.blocks * 8))
                            else:
                                checks.append(all(ctypes.string_at(ptr + w * size + (i + 1) * block - 8, 8)
                                    == bytes([pattern]) * 8 for i in range(args.blocks)))
                        if all(checks) or time.perf_counter_ns() - start > 10e9:
                            break
                        time.sleep(.001)
                    notice = {'tail_ok': all(checks), 'tail_attempts': attempts,
                              'tail_wait_ms': (time.perf_counter_ns() - start) / 1e6,
                              'receiver_threads_before_verify': thread_delta(active['before'], threads())}
                    active['tail_complete'] = notice['tail_ok']
                    self.reply(notice)
                elif self.path == '/verify':
                    assert active and active['round'] == data['round']
                    assert active.get('tail_complete'), 'observe all layer tails first'
                    cpu = thread_delta(active['before'], threads())
                    start = time.perf_counter_ns()
                    attempts = 0
                    while True:
                        attempts += 1
                        if args.memory == 'cuda':
                            arr = buf.cpu().numpy()
                        else:
                            arr = np.ctypeslib.as_array(buf).view(np.uint8)
                        checks = [bool(np.all(arr[w * size:(w + 1) * size] == pattern))
                                  for w, pattern in enumerate(active['patterns'])]
                        break  # Every-byte verification must pass on its first try.
                    record = {'round': active['round'], 'ok': all(checks), 'checks': checks,
                              'verify_attempts': attempts,
                              'verify_ms': (time.perf_counter_ns() - start) / 1e6,
                              'receiver_threads_before_verify': cpu}
                    rows.append(record)
                    save(args.root / 'receiver-rounds.json', rows)
                    # Never reuse a destination unless every byte was checked.
                    if record['ok']:
                        active.clear()
                    self.reply(record)
                elif self.path == '/stop':
                    self.reply({'stopped': True})
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    raise ValueError('unknown endpoint')
            except Exception as exc:
                save(args.root / 'receiver-failure.json', {'error': repr(exc)})
                self.send_error(500, repr(exc))

    server = http.server.HTTPServer((args.local_ip, args.http_port), Handler)
    save(args.root / 'receiver-meta.json', meta)
    print(json.dumps(meta), flush=True)
    try:
        server.serve_forever(poll_interval=.05)
    finally:
        server.server_close()
        engine.unregister_memory(ptr)


def sender(args):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, data=None):
        r = urllib.request.Request(args.server_url + path,
              data=json.dumps(data).encode() if data is not None else None,
              headers={'Content-Type': 'application/json'})
        with opener.open(r, timeout=20) as response:
            return json.load(response)

    meta = request('/')
    size = args.blocks * args.block_mib * 1024 * 1024
    assert (meta['bytes_per_writer'], meta['writers'], meta['blocks']) == (size, args.writers, args.blocks)
    assert 'MC_TCP_WRITE_REMOTE_FENCE' not in os.environ, 'this test matches the archival no-fence path'
    buffers = [ctypes.create_string_buffer(size) for _ in range(args.writers)]
    pointers = [ctypes.addressof(buf) for buf in buffers]
    engine, identity = initialize(args.local_ip)
    for ptr in pointers:
        assert engine.register_memory(ptr, size) == 0
    save(args.root / 'sender-meta.json', identity)
    block = args.block_mib * 1024 * 1024
    result = {'sender': identity, 'receiver': meta, 'writers': args.writers,
              'bytes_per_writer': size, 'warmup_rounds': args.warmup,
              'note': 'API duration is local completion, not last wire ACK or remote HBM completion',
              'complete': False, 'rounds': []}
    save(args.root / 'result.json', result)
    if args.start_delay:
        time.sleep(args.start_delay)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.writers) as pool:
            for round_id in range(args.warmup + args.iterations):
                patterns = [1 + (round_id * args.writers + w) % 250 for w in range(args.writers)]
                for ptr, pattern in zip(pointers, patterns):
                    ctypes.memset(ptr, pattern, size)
                request('/prepare', {'round': round_id, 'patterns': patterns})
                barrier = threading.Barrier(args.writers + 1)

                def transfer(writer):
                    source = [pointers[writer] + i * block for i in range(args.blocks)]
                    target = [meta['ptr'] + writer * size + i * block for i in range(args.blocks)]
                    barrier.wait(timeout=5)
                    start = time.perf_counter_ns()
                    rc = engine.batch_transfer_sync_write(meta['session'], source, target, [block] * args.blocks)
                    end = time.perf_counter_ns()
                    if rc != 0:
                        raise RuntimeError(f'writer {writer}: TE rc={rc}')
                    return {'writer': writer, 'tid': threading.get_native_id(),
                            'start_mono_ns': start, 'end_mono_ns': end, 'api_ms': (end - start) / 1e6, 'rc': rc}

                tcp_before, cpu_before = tcp_counters(), threads()
                futures = [pool.submit(transfer, w) for w in range(args.writers)]
                barrier.wait(timeout=5)
                writes = [f.result(timeout=45) for f in futures]
                cpu_after, tcp_after = threads(), tcp_counters()
                span_ms = (max(w['end_mono_ns'] for w in writes) - min(w['start_mono_ns'] for w in writes)) / 1e6
                notice = request('/await-complete', {'round': round_id})
                completion_notice_ms = (time.perf_counter_ns() - min(w['start_mono_ns'] for w in writes)) / 1e6
                if not notice['tail_ok']:
                    raise RuntimeError('remote layer-tail completion timeout')
                verify = request('/verify', {'round': round_id})
                record = {'round': round_id, 'warmup': round_id < args.warmup,
                          'span_ms': span_ms, 'aggregate_gbps': size * args.writers * 8 / span_ms / 1e6,
                          'completion_notice_ms': completion_notice_ms,
                          'completion_notice_gbps': size * args.writers * 8 / completion_notice_ms / 1e6,
                          'tail_attempts': notice['tail_attempts'], 'tail_wait_ms': notice['tail_wait_ms'],
                          'writes': writes, 'sender_threads': thread_delta(cpu_before, cpu_after),
                          'sender_host_tcp_delta': {k: tcp_after[k] - v for k, v in tcp_before.items() if tcp_after[k] != v},
                          **verify, 'receiver_threads_before_verify': notice['receiver_threads_before_verify']}
                result['rounds'].append(record)
                save(args.root / 'result.json', result)
                print(json.dumps({k: record[k] for k in ('round', 'warmup', 'span_ms', 'aggregate_gbps', 'ok', 'verify_attempts')}), flush=True)
                if not verify['ok']:
                    raise RuntimeError('remote full-buffer verification failed; stop load')
                time.sleep(args.idle_ms / 1000)
        measured = [r for r in result['rounds'] if not r['warmup']]
        result.update(complete=True, mean_span_ms=statistics.mean(r['span_ms'] for r in measured),
                      median_span_ms=statistics.median(r['span_ms'] for r in measured),
                      aggregate_gbps=size * args.writers * 8 * len(measured) / sum(r['span_ms'] for r in measured) / 1e6,
                      mean_writer_api_ms=statistics.mean(w['api_ms'] for r in measured for w in r['writes']),
                      completion_notice_gbps=size * args.writers * 8 * len(measured)
                        / sum(r['completion_notice_ms'] for r in measured) / 1e6,
                      mean_completion_notice_ms=statistics.mean(r['completion_notice_ms'] for r in measured))
    except BaseException as exc:
        result['failure'] = repr(exc)
        raise
    finally:
        save(args.root / 'result.json', result)
        # Stop only this diagnostic receiver. Never touch the model experiment.
        try:
            request('/stop', {})
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['receiver', 'sender'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--local-ip', required=True)
    parser.add_argument('--server-url', default='http://192.168.89.8:35876')
    parser.add_argument('--http-port', type=int, default=35876)
    parser.add_argument('--memory', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--writers', type=int, choices=[1, 4], default=1)
    parser.add_argument('--blocks', type=int, default=30)
    parser.add_argument('--block-mib', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--idle-ms', type=int, default=300)
    parser.add_argument('--start-delay', type=float, default=0, help='Attach external PID probes before load')
    args = parser.parse_args()
    assert 0 < args.blocks * args.block_mib * args.writers <= 1200
    assert 0 <= args.warmup <= 4 and 1 <= args.iterations <= 16
    assert 0 <= args.idle_ms <= 1000 and 0 <= args.start_delay <= 30
    args.root.mkdir(parents=True, exist_ok=True)
    assert not any((args.root / n).exists() for n in ('result.json', 'receiver-meta.json')), 'preserve previous attempt'
    (receiver if args.role == 'receiver' else sender)(args)


if __name__ == '__main__':
    main()
