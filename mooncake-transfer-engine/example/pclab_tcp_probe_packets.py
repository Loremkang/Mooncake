#!/usr/bin/env python3
"""Offline header-only TCP analysis. Record boundaries are not wire packets."""
import argparse
import json
from pathlib import Path
import socket
import struct


def packets(path):
    with open(path, 'rb', buffering=4*1024*1024) as f:
        magic = f.read(4)
        assert magic in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4',
                         b'\x4d\x3c\xb2\xa1', b'\xa1\xb2\x3c\x4d')
        endian = '<' if magic[0] in (0xd4, 0x4d) else '>'
        unit = 1e9 if magic in (b'\x4d\x3c\xb2\xa1', b'\xa1\xb2\x3c\x4d') else 1e6
        link = struct.unpack(endian+'HHIIII', f.read(20))[-1]
        assert link == 1, ('expected Ethernet', link)
        record = struct.Struct(endian+'IIII')
        while h := f.read(16):
            sec, frac, captured, original = record.unpack(h)
            p = f.read(captured)
            off = 14
            if len(p) < off+40:
                continue
            ether = struct.unpack_from('!H', p, 12)[0]
            while ether in (0x8100, 0x88a8):
                ether = struct.unpack_from('!H',p,off+2)[0]; off += 4
            if ether != 0x0800 or p[off+9] != 6:
                continue
            ihl = (p[off]&15)*4; t = off+ihl
            if len(p)<t+20: continue
            sp,dp,seq,ack=struct.unpack_from('!HHII',p,t)
            thl=(p[t+12]>>4)*4
            n=struct.unpack_from('!H',p,off+2)[0]-ihl-thl
            opts={}; j=t+20
            while j<min(t+thl,len(p)):
                kind=p[j]
                if kind==0: break
                if kind==1: j+=1; continue
                length=p[j+1]
                if length<2 or j+length>len(p): break
                if kind==8 and length==10: opts['ts']=struct.unpack_from('!II',p,j+2)
                if kind==5: opts['sack']=list(struct.unpack('!'+'I'*((length-2)//4),p[j+2:j+length]))
                if kind==3: opts['wscale']=p[j+2]
                j+=length
            yield {'t':sec+frac/unit,'src':socket.inet_ntoa(p[off+12:off+16]),
                   'sp':sp,'dp':dp,'seq':seq,'ack':ack,'n':max(0,n),
                   'flags':p[t+13],'win':struct.unpack_from('!H',p,t+14)[0],**opts}


def insert(ranges, start, end):
    overlap=sum(max(0,min(end,b)-max(start,a)) for a,b in ranges)
    # Typically one contiguous range; holes only expand this short list.
    result=[]
    for a,b in sorted(ranges+[(start,end)]):
        if result and a<=result[-1][1]: result[-1][1]=max(result[-1][1],b)
        else: result.append([a,b])
    ranges[:]=result
    return overlap


def scan(path, receiver_ip, bulk_bytes):
    flows={}; current={}; records=0; orphans=0
    for p in packets(path):
        records+=1; forward=p['src']!=receiver_ip
        pair=(p['sp'],p['dp']) if forward else (p['dp'],p['sp'])
        if forward and p['flags']&2 and not p['flags']&16:
            key=f'{pair[0]}:{pair[1]}:{p["seq"]}'
            current[pair]=key
            if key not in flows:
                flows[key]={'key':key,'isn':p['seq'],'syn':p['t'],'syn_count':0,
                    'ranges':[], 'data_records':0,'overlap_records':0,'overlap_bytes':0,
                    'out_of_order_records':0,'overlaps':[],'gaps':[],
                    'zero_window_records':0,'sack_records':0,'rst_records':0}
            flows[key]['syn_count']+=1
        key=current.get(pair)
        if key is None: orphans+=1; continue
        v=flows[key]
        if not forward and p['flags']&2: v.setdefault('synack',p['t'])
        if p['flags']&4: v['rst_records']+=1
        if forward and p['flags']&1: v.setdefault('fin',p['t'])
        if forward and p['n']:
            start=(p['seq']-v['isn']-1)&0xffffffff; end=start+p['n']
            if start>1<<30: continue
            prior=v['ranges'][0][1] if v['ranges'] and v['ranges'][0][0]==0 else 0
            if start>prior: v['out_of_order_records']+=1
            if 'last_data_t' in v and p['t']-v['last_data_t']>.05 and prior<bulk_bytes:
                v['gaps'].append({'ms':(p['t']-v['last_data_t'])*1000,
                    'start_t':v['last_data_t'],'end_t':p['t'],'offset':start,'prior_contiguous':prior})
            overlap=insert(v['ranges'],start,end)
            if overlap:
                v['overlap_records']+=1; v['overlap_bytes']+=overlap
                if len(v['overlaps'])<80:
                    v['overlaps'].append({'t':p['t'],'offset':start,'length':p['n'],'ts':p.get('ts')})
            v['data_records']+=1; v['last_data_t']=p['t']
            v.setdefault('first_data',p['t'])
            if v['ranges'][0][0]==0 and v['ranges'][0][1]>=bulk_bytes:
                v.setdefault('bulk_complete',p['t'])
        if not forward:
            ack=(p['ack']-v['isn']-1)&0xffffffff
            if p['win']==0: v['zero_window_records']+=1
            if p.get('sack'): v['sack_records']+=1
            if ack<1<<30:
                if ack>=bulk_bytes: v.setdefault('bulk_ack',p['t'])
                if ack>v.get('max_ack',0):
                    v['max_ack']=ack; v['last_ack_progress']=p['t']
            if p['n']==1: v.setdefault('fence_response',p['t'])
    return {'records':records,'orphans':orphans,'flows':flows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True);ap.add_argument('--receiver',required=True)
    ap.add_argument('--result',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--receiver-ip',default='192.168.89.8')
    a=ap.parse_args(); result=json.loads(Path(a.result).read_text())
    bulk=result['bytes_per_iteration']//result['receiver']['blocks']+24
    s=scan(a.source,a.receiver_ip,bulk);r=scan(a.receiver,a.receiver_ip,bulk)
    output={'source':s,'receiver':r,'batches':[],'bulk_tcp_bytes':bulk}
    for row in result['rows']:
        t0=row['sender_start_real_ns']/1e9;t1=row['sender_end_real_ns']/1e9
        group=[v for v in s['flows'].values() if t0<=v['syn']<=t1]
        if not group:
            output['batches'].append({'iteration':row['iteration'],'missing':True});continue
        critical=max(group,key=lambda v:v.get('fence_response',v.get('fin',v['syn'])))
        peer=r['flows'].get(critical['key'],{})
        def delta(v,end,start):
            return (v[end]-v[start])*1000 if end in v and start in v else None
        bounds=None
        if 'synack' in critical and 'synack' in peer:
            bounds=[(peer['synack']-critical['synack'])*1000,
                    (peer['syn']-critical['syn'])*1000]
        output['batches'].append({'iteration':row['iteration'],'warmup':row['warmup'],
            'api_ms':row['api_ms'],'connections':len(group),
            'syn_spread_ms':(max(v['syn'] for v in group)-min(v['syn'] for v in group))*1000,
            'max_handshake_ms':max(delta(v,'synack','syn') or 0 for v in group),
            'source_overlap_bytes':sum(v['overlap_bytes'] for v in group),
            'source_zero_window_records':sum(v['zero_window_records'] for v in group),
            'critical':{'key':critical['key'],'handshake_ms':delta(critical,'synack','syn'),
                'syn_to_source_bulk_complete_ms':delta(critical,'bulk_complete','syn'),
                'syn_to_source_bulk_ack_ms':delta(critical,'bulk_ack','syn'),
                'syn_to_fence_ms':delta(critical,'fence_response','syn'),
                'source_bulk_ack_to_fence_ms':delta(critical,'fence_response','bulk_ack'),
                'receiver_syn_to_bulk_complete_ms':delta(peer,'bulk_complete','syn'),
                'receiver_bulk_complete_to_fence_ms':delta(peer,'fence_response','bulk_complete'),
                'receiver_minus_source_clock_bounds_ms':bounds,
                'source_overlap_bytes':critical['overlap_bytes'],
                'receiver_overlap_bytes':peer.get('overlap_bytes'),
                'source_gaps':sorted(critical['gaps'],key=lambda v:-v['ms'])[:5],
                'receiver_gaps':sorted(peer.get('gaps',[]),key=lambda v:-v['ms'])[:5]}})
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'batches':output['batches'],
        'source_records':s['records'],'receiver_records':r['records'],
        'source_orphans':s['orphans'],'receiver_orphans':r['orphans']}))


if __name__=='__main__': main()
