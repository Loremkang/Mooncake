#!/usr/bin/env python3
"""Read retained native A/B artifacts on receiver. Never starts traffic."""
import bisect,collections,itertools,json,pathlib,re,statistics,struct,sys
root=pathlib.Path(sys.argv[1]);PORT=35881

def stats(v):
 v=sorted(v)
 return dict(n=len(v),min=v[0],p50=statistics.median(v),p95=v[min(len(v)-1,int(.95*len(v)))],max=v[-1]) if v else {}

def packets(path,ports):
 out=collections.defaultdict(list)
 with open(path,'rb') as f:
  h=f.read(24);assert h[:4]==b'\xd4\xc3\xb2\xa1' and struct.unpack_from('<I',h,20)[0]==1
  while h:=f.read(16):
   sec,us,n,_=struct.unpack('<IIII',h);p=f.read(n)
   if len(p)<54 or p[12:14]!=b'\x08\x00' or p[23]!=6:continue
   ihl=(p[14]&15)*4;t=14+ihl;sp,dp,seq,ack=struct.unpack_from('!HHII',p,t)
   forward=dp==PORT;port=sp if forward else dp
   if port not in ports or (sp!=PORT and dp!=PORT):continue
   thl=(p[t+12]>>4)*4;ln=struct.unpack_from('!H',p,16)[0]-ihl-thl;tv=te=None;j=t+20
   while j<min(t+thl,len(p)):
    kind=p[j]
    if kind==0:break
    if kind==1:j+=1;continue
    size=p[j+1]
    if size<2 or j+size>len(p):break
    if kind==8 and size==10:tv,te=struct.unpack_from('!II',p,j+2)
    j+=size
   # epoch_us, direction, seq, ack, payload bytes, flags, TSval, TSecr
   out[port].append((sec*1000000+us,int(forward),seq,ack,ln,p[t+13],tv,te))
 return out

def host_samples(path):
 out=[];s=None
 with open(path) as f:
  for line in f:
   if re.fullmatch(r'\d{10}\.\d+\n',line):
    if s:out.append(s)
    s=dict(t=float(line),cpu={},softnet=[],sockets=[]);continue
   if s is None:continue
   a=line.split()
   if not a:continue
   if a[0].startswith('cpu') and len(a)>8:s['cpu'][a[0]]=[int(x) for x in a[1:9]]
   elif re.match(r'^[0-9a-f]{8}(?:\s+[0-9a-f]{8}){8}',line):s['softnet'].append([int(v,16) for v in a[:3]])
   elif a[0] in ['ESTAB','SYN-SENT','FIN-WAIT-1','FIN-WAIT-2','CLOSE-WAIT','LAST-ACK']:
    detail=next(f,''); tokens=dict(re.findall(r'\b([a-zA-Z][a-zA-Z_]*):([^\s]+)',detail))
    s['sockets'].append(dict(state=a[0],recvq=int(a[1]),sendq=int(a[2]),endpoints=a[3:5],**tokens))
 if s:out.append(s)
 return out

def host_stats(samples,lo,hi):
 v=[s for s in samples if lo<=s['t']<=hi and s['cpu']]
 if len(v)<2:return {}
 total=collections.defaultdict(lambda:[0]*8);maxsoft=maxbusy=0
 for a,b in zip(v,v[1:]):
  for k,x in a['cpu'].items():
   if k not in b['cpu']:continue
   d=[y-z for z,y in zip(x,b['cpu'][k])];n=sum(d)
   if n<=0:continue
   for i,x in enumerate(d):total[k][i]+=x
   if k!='cpu':maxsoft=max(maxsoft,100*d[6]/n);maxbusy=max(maxbusy,100*(n-d[3]-d[4])/n)
 g=total['cpu'];n=sum(g)
 soft=lambda x:[sum(v[i] for v in x['softnet']) for i in range(3)]
 return dict(samples=len(v),global_busy_pct=100*(n-g[3]-g[4])/n,global_steal_pct=100*g[7]/n,max_core_sample_softirq_pct=maxsoft,max_core_sample_busy_pct=maxbusy,softnet_delta=[y-x for x,y in zip(soft(v[0]),soft(v[-1]))])

def socket_stats(samples,port,lo,hi):
 rows=[];wanted=['rtt','cwnd','unacked','notsent','retrans','snd_wnd','rwnd_limited','sndbuf_limited','bytes_sent','bytes_acked','bytes_received']
 for s in samples:
  if not lo<=s['t']<=hi:continue
  for x in s['sockets']:
   if not any(v.endswith(':'+str(port)) for v in x['endpoints']):continue
   rows.append(dict(t=s['t'],recvq=x['recvq'],sendq=x['sendq'],**{k:x[k] for k in wanted if k in x}))
 rt=lambda x:float(x.get('rtt','0').split('/')[0])
 return dict(samples=len(rows),rtt_ms=stats([rt(x) for x in rows]),cwnd=stats([int(x['cwnd']) for x in rows if 'cwnd' in x]),max_notsent=max((int(x.get('notsent',0)) for x in rows),default=0),max_recvq=max((x['recvq'] for x in rows),default=0),max_sendq=max((x['sendq'] for x in rows),default=0),rwnd_limited_samples=sum('rwnd_limited' in x for x in rows),highest_rtt=sorted(rows,key=rt,reverse=True)[:3])

meta={};allports=set()
for case in ['a1','b1','b2','a2']:
 rows=[json.loads(x) for x in (root/'source'/case/'client.jsonl').read_text().splitlines()];measured=[x for x in rows[:-1] if not x['warmup']]
 median=statistics.median(x['ms'] for x in measured)
 selected={'slowest':max(measured,key=lambda x:x['ms']),'median':min(measured,key=lambda x:abs(x['ms']-median))}
 for r in selected.values():r['critical']=max(r['flows'],key=lambda x:x['ack_ms']);allports.add(r['critical']['port'])
 meta[case]=dict(rows=rows,selected=selected)
rx=packets(root/'receiver.pcap',allports);rsamples=host_samples(root/'host.log');out={}
for case,meta_case in meta.items():
 ps={r['critical']['port'] for r in meta_case['selected'].values()};tx=packets(root/'source'/case/'source.pcap',ps);ssamples=host_samples(root/'source'/case/'host.log');res={}
 for label,row in meta_case['selected'].items():
  port=row['critical']['port'];lo=row['epoch_ns']/1000;hi=lo+row['ms']*1000
  # Include only this batch's source-side records. Unique full TCP header matches
  # exclude captures where retransmitted headers cannot be distinguished.
  sx=[p for p in tx[port] if lo-1000<=p[0]<=hi+1000];sm=collections.defaultdict(list);rm=collections.defaultdict(list)
  for p in sx:sm[p[1:]].append(p[0])
  for p in rx[port]:rm[p[1:]].append(p[0])
  pairs=[(st[0],rm[k][0],k) for k,st in sm.items() if len(st)==1 and len(rm.get(k,[]))==1]
  forward=[r-s for s,r,k in pairs if k[0]];reverse=[r-s for s,r,k in pairs if not k[0]]
  clock_lo=max(reverse);clock_hi=min(forward);assert clock_lo<=clock_hi,(case,label,clock_lo,clock_hi)
  def ds(direction,mss=False):
   d=[r-s if direction else s-r for s,r,k in pairs if bool(k[0])==direction and (k[3]==1448 if mss else k[3]>0 if direction else True)]
   low=[(x-clock_hi if direction else x+clock_lo)/1000 for x in d];high=[(x-clock_lo if direction else x+clock_hi)/1000 for x in d]
   return dict(lower_ms=stats(low),upper_ms=stats(high))
  data=[(s,r,k) for s,r,k in pairs if k[0] and k[3]>0]
  # Report observed records, not a wire-packet loss/reordering probability.
  late=0;earliest=2**63
  for st,group in itertools.groupby(sorted(data,key=lambda x:-x[0]),key=lambda x:x[0]):
   group=list(group);late+=sum(r>earliest for s,r,k in group);earliest=min(earliest,min(r for s,r,k in group))
  offset=(clock_lo+clock_hi)/2e6
  res[label]=dict(batch=row['batch'],batch_ms=row['ms'],port=port,clock_offset_bounds_ms=[clock_lo/1000,clock_hi/1000],source_records=len(sx),unique_matches=len(pairs),matched_forward_records=len(data),forward=ds(True),forward_exact1448=ds(True,True),reverse=ds(False),matched_forward_late_records=late,source_socket=socket_stats(ssamples,port,lo/1e6,hi/1e6),receiver_socket=socket_stats(rsamples,port,lo/1e6+offset,hi/1e6+offset),largest_exact1448=sorted([dict(source_epoch_us=s,receiver_epoch_us=r,seq=k[1],len=k[3],delay_bounds_ms=[(r-s-clock_hi)/1000,(r-s-clock_lo)/1000]) for s,r,k in data if k[3]==1448],key=lambda x:x['delay_bounds_ms'][0],reverse=True)[:2])
 rows=meta_case['rows'][:-1];a=rows[0]['epoch_ns']/1e9;b=rows[-1]['epoch_ns']/1e9+rows[-1]['ms']/1000;offset=sum(res['slowest']['clock_offset_bounds_ms'])/2000
 res['cpu_source']=host_stats(ssamples,a,b);res['cpu_receiver']=host_stats(rsamples,a+offset,b+offset)
 out[case]=res
(root/'packet-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
