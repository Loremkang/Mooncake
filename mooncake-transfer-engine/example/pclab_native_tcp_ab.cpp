// Native RAM TCP A/B: one poll thread per endpoint; no Mooncake/CUDA dependency.
// Every frame is fully checked before its header is echoed as an application ACK.
// Build: g++ -O2 -std=c++17 -Wall -Wextra pclab_native_tcp_ab.cpp -o native-ab
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
using Bytes = std::array<unsigned char, 32>;
constexpr size_t CHUNK=65536;
volatile sig_atomic_t stopped=0;
void on_signal(int) { stopped=1; }
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
long long epoch() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
void require(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
bool again() { return errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR; }
void put(Bytes& h,int p,uint64_t v) { for(int i=7;i>=0;--i) { h[p+i]=v&255; v>>=8; } }
uint64_t get(const Bytes& h,int p) { uint64_t v=0; for(int i=0;i<8;++i) v=(v<<8)|h[p+i]; return v; }
Bytes header(uint64_t n,uint64_t batch,uint64_t flow) { Bytes h{}; put(h,0,0x50434c4142544350ULL); put(h,8,n); put(h,16,batch); put(h,24,flow); return h; }
unsigned char pattern(const Bytes& h) { return (get(h,16)*37+get(h,24)*17)%251+1; }
sockaddr_in addr(const std::string& ip,int port) { sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(port); require(inet_pton(AF_INET,ip.c_str(),&a.sin_addr)==1,"IPv4 address"); return a; }
void config(int fd) { int one=1; require(setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one))==0,"TCP_NODELAY"); require(fcntl(fd,F_SETFL,O_NONBLOCK)==0,"nonblocking"); }
int port_of(int fd) { sockaddr_in a{}; socklen_t n=sizeof(a); require(getsockname(fd,(sockaddr*)&a,&n)==0,"getsockname"); return ntohs(a.sin_port); }
uint32_t retrans(int fd) {
#ifdef __linux__
  tcp_info t{}; socklen_t n=sizeof(t); require(getsockopt(fd,IPPROTO_TCP,TCP_INFO,&t,&n)==0,"TCP_INFO"); return t.tcpi_total_retrans;
#else
  (void)fd; return 0;
#endif
}
void cpu() { rusage u{}; getrusage(RUSAGE_SELF,&u); std::cout << ",\"cpu_user_s\":" << u.ru_utime.tv_sec+u.ru_utime.tv_usec/1e6 << ",\"cpu_sys_s\":" << u.ru_stime.tv_sec+u.ru_stime.tv_usec/1e6; }
struct Receiver { Bytes h{}; size_t hn=0,an=0; uint64_t body=0; std::array<unsigned char,CHUNK> expect{}; };
void server(const std::string& ip,int port,int duration) {
  int listener=socket(AF_INET,SOCK_STREAM,0); require(listener>=0,"socket"); config(listener);
  int one=1; setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one)); auto a=addr(ip,port);
  require(bind(listener,(sockaddr*)&a,sizeof(a))==0,"bind"); require(listen(listener,128)==0,"listen");
  std::map<int,Receiver> connections; std::array<unsigned char,CHUNK> buf{};
  uint64_t frames=0,bytes=0,accepted=0,closed=0; auto start=Clock::now();
  std::cout << "{\"event\":\"ready\",\"epoch_ns\":"<<epoch()<<",\"pid\":"<<getpid()<<"}"<<std::endl;
  while(!stopped && ms(start)<duration*1000.) {
    std::vector<pollfd> pf{{listener,POLLIN,0}};
    for(auto& [fd,c]:connections) pf.push_back({fd,(short)((c.hn==32 && c.body==get(c.h,8))?POLLOUT:POLLIN),0});
    int n=poll(pf.data(),pf.size(),100); if(n<0) { if(errno==EINTR) continue; throw std::runtime_error("poll"); }
    if(pf[0].revents&POLLIN) {
      for(;;) { int fd=accept(listener,nullptr,nullptr); if(fd<0) { require(again(),"accept"); break; } config(fd); connections.emplace(fd,Receiver{}); ++accepted; }
    }
    for(size_t i=1;i<pf.size();++i) {
      int fd=pf[i].fd; auto& c=connections.at(fd); bool remove=false;
      require(!(pf[i].revents&(POLLERR|POLLNVAL)),"receiver socket error");
      if(pf[i].revents&(POLLIN|POLLHUP)) {
        if(c.hn<32) {
          auto z=recv(fd,c.h.data()+c.hn,32-c.hn,0);
          if(z==0) { require(c.hn==0,"partial header EOF"); remove=true; }
          else if(z<0) require(again(),"recv header");
          else { c.hn+=z; if(c.hn==32) { require(get(c.h,0)==0x50434c4142544350ULL && get(c.h,8)>0 && get(c.h,8)<=(1ULL<<30),"invalid header"); c.expect.fill(pattern(c.h)); } }
        } else if(c.body<get(c.h,8)) {
          size_t need=std::min<uint64_t>(CHUNK,get(c.h,8)-c.body); auto z=recv(fd,buf.data(),need,0);
          if(z<0) require(again(),"recv payload");
          else { require(z>0,"partial payload EOF"); require(memcmp(buf.data(),c.expect.data(),z)==0,"payload mismatch"); c.body+=z; }
        }
      }
      if(!remove && (pf[i].revents&POLLOUT)) {
        auto z=send(fd,c.h.data()+c.an,32-c.an,0);
        if(z<0) require(again(),"send ack");
        else { require(z>0,"zero send"); c.an+=z; if(c.an==32) { ++frames; bytes+=c.body; c.hn=c.an=0; c.body=0; } }
      }
      if(remove) { close(fd); connections.erase(fd); ++closed; }
    }
  }
  bool partial=false; for(auto& [fd,c]:connections) { partial|=c.hn!=0; close(fd); } close(listener);
  std::cout<<"{\"event\":\"server_summary\",\"ok\":"<<(partial?"false":"true")<<",\"frames\":"<<frames<<",\"bytes\":"<<bytes<<",\"accepted\":"<<accepted<<",\"closed\":"<<closed<<",\"open_at_stop\":"<<connections.size(); cpu(); std::cout<<"}"<<std::endl;
  require(!partial,"server stopped mid-frame");
}
struct Sender { int fd=-1,port=0; bool connecting=false,done=false; Bytes h{},ack{}; size_t hn=0,an=0; uint64_t body=0; std::array<unsigned char,CHUNK> buf{}; uint32_t r0=0; double connect_ms=0,send_ms=0,ack_ms=0; };
void client(const std::string& ip,int port,const std::string& local,const std::string& mode,int flows,uint64_t bytes,int batches,int warmup,int idle_ms,int batch_base) {
  require(mode=="short" || mode=="reuse","mode");
  std::vector<Sender> cs(flows); auto dest=addr(ip,port); uint64_t total=0;
  for(int batch=0;batch<batches;++batch) {
    for(int i=0;i<flows;++i) { auto& c=cs[i]; c.h=header(bytes,batch+batch_base,i); c.ack={}; c.buf.fill(pattern(c.h)); c.hn=c.an=c.body=0; c.done=false; c.send_ms=c.ack_ms=c.connect_ms=0; }
    auto t=Clock::now(); auto en=epoch();
    for(auto& c:cs) {
      if(c.fd<0) {
        c.fd=socket(AF_INET,SOCK_STREAM,0); require(c.fd>=0,"socket"); config(c.fd); auto la=addr(local,0);
        require(bind(c.fd,(sockaddr*)&la,sizeof(la))==0,"source bind");
        int z=connect(c.fd,(sockaddr*)&dest,sizeof(dest)); require(z==0 || errno==EINPROGRESS,"connect"); c.connecting=z!=0; c.port=port_of(c.fd);
      }
      c.r0=retrans(c.fd);
    }
    int remaining=flows;
    while(remaining) {
      require(!stopped,"interrupted"); require(ms(t)<10000,"batch exceeded 10 seconds");
      std::vector<pollfd> pf; for(auto& c:cs) pf.push_back({c.fd,(short)(c.done?0:(c.connecting || c.hn<32 || c.body<bytes)?POLLOUT:POLLIN),0});
      int n=poll(pf.data(),pf.size(),100); if(n<0) { if(errno==EINTR) continue; throw std::runtime_error("client poll"); }
      for(int i=0;i<flows;++i) {
        auto& c=cs[i]; auto ev=pf[i].revents; if(c.done) continue;
        require(!(ev&(POLLERR|POLLNVAL)),"client socket error");
        if(c.connecting && (ev&POLLOUT)) { int e=0; socklen_t l=sizeof(e); require(getsockopt(c.fd,SOL_SOCKET,SO_ERROR,&e,&l)==0 && e==0,"connect completion"); c.connecting=false; c.connect_ms=ms(t); }
        if(!c.connecting && (ev&POLLOUT)) {
          if(c.hn<32) { auto z=send(c.fd,c.h.data()+c.hn,32-c.hn,0); if(z<0) require(again(),"send header"); else { require(z>0,"zero send"); c.hn+=z; } }
          else if(c.body<bytes) { auto z=send(c.fd,c.buf.data(),std::min<uint64_t>(CHUNK,bytes-c.body),0); if(z<0) require(again(),"send payload"); else { require(z>0,"zero send"); c.body+=z; if(c.body==bytes)c.send_ms=ms(t); } }
        }
        if(ev&(POLLIN|POLLHUP)) { auto z=recv(c.fd,c.ack.data()+c.an,32-c.an,0); if(z<0) require(again(),"recv ack"); else { require(z>0,"ACK EOF"); c.an+=z; if(c.an==32) { require(c.ack==c.h,"ACK mismatch"); c.ack_ms=ms(t); c.done=true; --remaining; } } }
      }
    }
    double elapsed=ms(t); total+=bytes*flows;
    std::cout<<"{\"event\":\"batch\",\"mode\":\""<<mode<<"\",\"batch\":"<<batch<<",\"wire_batch\":"<<batch+batch_base<<",\"warmup\":"<<(batch<warmup?"true":"false")<<",\"epoch_ns\":"<<en<<",\"ms\":"<<elapsed<<",\"gbps\":"<<bytes*flows*8./(elapsed*1e6)<<",\"bytes\":"<<bytes*flows<<",\"flows\":[";
    for(int i=0;i<flows;++i) { auto& c=cs[i]; if(i)std::cout<<','; std::cout<<"{\"port\":"<<c.port<<",\"connect_ms\":"<<c.connect_ms<<",\"send_ms\":"<<c.send_ms<<",\"ack_ms\":"<<c.ack_ms<<",\"retrans_delta\":"<<retrans(c.fd)-c.r0<<"}"; if(mode=="short") { close(c.fd); c.fd=-1; } }
    std::cout<<"]}"<<std::endl;
    if(batch+1<batches)std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
  }
  for(auto& c:cs)if(c.fd>=0)close(c.fd);
  std::cout<<"{\"event\":\"client_summary\",\"ok\":true,\"bytes\":"<<total<<",\"frames\":"<<flows*batches;cpu();std::cout<<"}"<<std::endl;
}
int main(int argc,char** argv) {
  signal(SIGPIPE,SIG_IGN); signal(SIGINT,on_signal); signal(SIGTERM,on_signal); std::cout.precision(12);
  try {
    require(argc>=2,"server IP PORT DURATION | client IP PORT LOCAL MODE FLOWS BYTES BATCHES WARMUP IDLE_MS BATCH_BASE");
    if(std::string(argv[1])=="server") { require(argc==5,"server args"); server(argv[2],std::stoi(argv[3]),std::stoi(argv[4])); }
    else { require(std::string(argv[1])=="client" && argc==12,"client args"); int f=std::stoi(argv[6]),b=std::stoi(argv[8]); auto n=std::stoull(argv[7]); require(f>0 && f<=128 && b>0 && b<=1000 && n>0 && n<=(1ULL<<30),"bounds"); client(argv[2],std::stoi(argv[3]),argv[4],argv[5],f,n,b,std::stoi(argv[9]),std::stoi(argv[10]),std::stoi(argv[11])); }
    return 0;
  } catch(const std::exception& e) { std::cerr<<"{\"event\":\"failure\",\"epoch_ns\":"<<epoch()<<",\"reason\":\""<<e.what()<<"\",\"errno\":"<<errno<<"}"<<std::endl; return 2; }
}
