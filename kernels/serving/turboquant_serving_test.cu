/**
 * @file
 * @brief Harness for the TurboQuant serving kernels (turboquant_kernels.cuh).
 *
 * The store kernels and the fp8-key full-dequant are checked by byte-exact
 * host replay: every op on those paths (cvt.rz/cvt.rn f16 conversion, integer
 * packing, IEEE fma) is emulated exactly, and the two div.full.f32 spots are
 * evaluated on-device through a one-element helper kernel — IEEE division is
 * not a stand-in there, because quantization steps of coarse (bf16) inputs
 * land exactly on x.5 rounding boundaries where div.full's <=2 ulp deviation
 * flips the code. The split-KV decode uses
 * ex2.approx and div.full in ways no host replay can reproduce, so stage1 +
 * stage2 are checked end-to-end against an fp64 softmax reference that
 * dequantizes the same cache bytes (fp8 and MSE-norm-corrected key paths,
 * GQA, sliding window, multi-split).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_80,code=sm_80 \
 *        turboquant_serving_test.cu -o turboquant_serving_test.out
 * Run: CUDA_VISIBLE_DEVICES=0 ./turboquant_serving_test.out
 */
#include "turboquant_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using namespace tms;
static int g_fail = 0;
#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
template <typename T> T* dnew(const std::vector<T>& h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template <typename T> T* dz(size_t n){T*d;CK(cudaMalloc(&d,n*sizeof(T)));CK(cudaMemset(d,0,n*sizeof(T)));return d;}
template <typename T> std::vector<T> d2h(const T*d,size_t n){std::vector<T> h(n);CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
static std::mt19937 rng(1207);
static std::vector<float> rv(size_t n,float lo,float hi){std::uniform_real_distribution<float> d(lo,hi);std::vector<float> v(n);for(auto&x:v)x=d(rng);return v;}
static void rep_ex(const char* nm,long mm){printf("%-34s %s (%ld mismatch)\n",nm,mm?"FAIL":"PASS",mm);if(mm)++g_fail;}
static void rep_rel(const char* nm,double e,double tol){printf("%-34s %s (rel %.3e)\n",nm,e<=tol?"PASS":"FAIL",e);if(e>tol)++g_fail;}

// ---- exact host f16 emulation ----
static float h2f(uint16_t h){
    uint32_t s=uint32_t(h&0x8000u)<<16; int e=(h>>10)&0x1F; uint32_t m=h&0x3FFu,x;
    if(e==0){ if(!m) x=s; else { int sh=0; while(!(m&0x400u)){m<<=1;++sh;} m&=0x3FFu; x=s|uint32_t(113-sh)<<23|(m<<13);} }
    else if(e==31) x=s|0x7F800000u|(m<<13);
    else x=s|uint32_t(e+112)<<23|(m<<13);
    float f; memcpy(&f,&x,4); return f;
}
static uint16_t f2h_rz(float f){  // cvt.rz.f16.f32: truncate toward zero
    uint32_t x; memcpy(&x,&f,4);
    uint16_t s=uint16_t((x>>16)&0x8000u); uint32_t ax=x&0x7FFFFFFFu;
    if(ax>0x7F800000u) return uint16_t(s|0x7E00u);
    if(ax==0x7F800000u) return uint16_t(s|0x7C00u);
    int e=int(ax>>23); uint32_t m=ax&0x7FFFFFu;
    if(e>142) return uint16_t(s|0x7BFFu);                      // overflow -> max finite
    if(e>=113) return uint16_t(s|uint32_t(e-112)<<10|(m>>13)); // normal
    if(e>=103) return uint16_t(s|((m|0x800000u)>>(126-e)));    // subnormal
    return s;
}
static uint16_t f2h_rn(float f){  // cvt.rn.f16.f32 / __float2half_rn: round-nearest-even
    uint32_t x; memcpy(&x,&f,4);
    uint16_t s=uint16_t((x>>16)&0x8000u); uint32_t ax=x&0x7FFFFFFFu;
    if(ax>0x7F800000u) return uint16_t(s|0x7E00u);
    if(ax>=0x47800000u) return uint16_t(s|0x7C00u);            // overflow -> inf
    int e=int(ax>>23); uint32_t m=ax&0x7FFFFFu;
    if(e>=113){
        uint32_t h=uint32_t(e-112)<<10|(m>>13), rem=m&0x1FFFu;
        if(rem>0x1000u||(rem==0x1000u&&(h&1u))) ++h;
        return uint16_t(s|h);
    }
    if(e>=102){
        int sh=126-e; uint32_t q=m|0x800000u, h=q>>sh, rem=q&((1u<<sh)-1u), half=1u<<(sh-1);
        if(rem>half||(rem==half&&(h&1u))) ++h;
        return uint16_t(s|h);
    }
    return s;
}
static void put16(uint8_t*p,uint16_t u){p[0]=uint8_t(u&0xFF);p[1]=uint8_t(u>>8);}
static uint16_t get16(const uint8_t*p){return uint16_t(p[0]|(p[1]<<8));}
static uint8_t ref_e4b15(float x){uint16_t h=f2h_rz(x);uint32_t s=h&0x8000u,a=h&0x7FFFu;if(a>0x3F00u)a=0x3F00u;return uint8_t((s|((a<<1)+0x80u))>>8);}
static uint16_t e4b15_f16(uint8_t b){return uint16_t(((b&0x80u)<<8)|((b&0x7Fu)<<7));}

// div.full.f32 oracle: the one op the host cannot emulate exactly
__global__ void k_div_full(const float* a,const float* b,float* c){ *c=tq_div(*a,*b); }
static float dev_div(float a,float b){
    static float* d=nullptr; if(!d) CK(cudaMalloc(&d,3*sizeof(float)));
    float h[2]={a,b}; CK(cudaMemcpy(d,h,2*sizeof(float),cudaMemcpyHostToDevice));
    k_div_full<<<1,1>>>(d,d+1,d+2);
    float r; CK(cudaMemcpy(&r,d+2,sizeof(float),cudaMemcpyDeviceToHost)); return r;
}

// ---- host replay of the shared value-quant tail ----
static void ref_value_q(const float* vals,uint8_t* cache,int64_t slot_base,int D,int kps,int vqb,int vdb){
    float vmin=INFINITY,vmax=-INFINITY;
    for(int d=0;d<D;++d){vmin=std::min(vmin,vals[d]);vmax=std::max(vmax,vals[d]);}
    const float v_scale=std::max(dev_div(vmax-vmin,vqb==3?7.0f:15.0f),1e-8f);
    const int qmax=vqb==3?7:15;
    std::vector<uint8_t> sq(D);
    for(int d=0;d<D;++d){int q=int(dev_div(vals[d]-vmin,v_scale)+0.5f);sq[d]=uint8_t(std::min(std::max(q,0),qmax));}
    const int64_t vb=slot_base+kps;
    if(vqb==4) for(int j=0;j<vdb;++j) cache[vb+j]=uint8_t((sq[2*j]&0xF)|((sq[2*j+1]&0xF)<<4));
    else for(int g=0;g<D/8;++g){uint32_t p=0;for(int i=0;i<8;++i)p|=uint32_t(sq[8*g+i]&0x7)<<(3*i);
        cache[vb+3*g]=uint8_t(p&0xFF);cache[vb+3*g+1]=uint8_t((p>>8)&0xFF);cache[vb+3*g+2]=uint8_t((p>>16)&0xFF);}
    put16(cache+vb+vdb,f2h_rn(v_scale)); put16(cache+vb+vdb+2,f2h_rn(vmin));
}
static float ref_value_idx(const uint8_t* cache,int64_t vb,int vqb,int d){
    if(vqb==4) return float((cache[vb+(d>>1)]>>((d&1)*4))&0xF);
    const int bit=d*3; return float(((cache[vb+(bit>>3)]|(cache[vb+(bit>>3)+1]<<8))>>(bit&7))&0x7);
}
static int ref_mse_idx(const uint8_t* cache,int64_t sb,int bits,int d){
    const int bit=d*bits; return ((cache[sb+(bit>>3)]|(cache[sb+(bit>>3)+1]<<8))>>(bit&7))&((1<<bits)-1);
}

// slot layout helpers
struct Layout{int D,H,bs,kps,vqb,vdb,slot;int64_t s_h,s_p,s_b;};
static Layout mk_layout(int D,int H,int bs,int kps,int vqb){
    Layout L{D,H,bs,kps,vqb,vqb==4?D/2:3*D/8,0,0,0,0};
    L.slot=L.kps+L.vdb+4; L.s_h=L.slot; L.s_p=int64_t(H)*L.slot; L.s_b=int64_t(bs)*L.s_p; return L;
}
static int64_t slot_base(const Layout&L,int64_t slot,int head){
    return (slot/L.bs)*L.s_b+(slot%L.bs)*L.s_p+int64_t(head)*L.s_h;
}

// ---- fp8-key store: kernel vs host replay ----
template <typename T>
static void test_store_fp8(const char* nm,int D,int vqb,bool bf16){
    const int N=12,H=2,BS=16,NB=6;
    Layout L=mk_layout(D,H,BS,D,vqb);
    std::vector<int> slots(N); std::vector<int> perm(NB*BS); for(int i=0;i<NB*BS;++i)perm[i]=i;
    std::shuffle(perm.begin(),perm.end(),rng);
    for(int t=0;t<N;++t) slots[t]=perm[t];
    slots[5]=-1;  // padding token
    auto kf=rv((size_t)N*H*D,-2,2), vf=rv((size_t)N*H*D,-2,2);
    std::vector<uint16_t> kb((size_t)N*H*D),vb((size_t)N*H*D);
    for(size_t i=0;i<kf.size();++i){
        if(bf16){uint32_t u;memcpy(&u,&kf[i],4);kb[i]=uint16_t(u>>16);memcpy(&u,&vf[i],4);vb[i]=uint16_t(u>>16);
            uint32_t r=uint32_t(kb[i])<<16;memcpy(&kf[i],&r,4);r=uint32_t(vb[i])<<16;memcpy(&vf[i],&r,4);}
        else {kb[i]=f2h_rn(kf[i]);vb[i]=f2h_rn(vf[i]);kf[i]=h2f(kb[i]);vf[i]=h2f(vb[i]);}
    }
    auto dk=dnew(kb); auto dv=dnew(vb); auto ds=dnew(slots);
    const size_t CB=(size_t)NB*BS*H*L.slot;
    uint8_t* dc=dz<uint8_t>(CB);
    tq_store_fp8<T,int><<<N*H,32>>>((const T*)dk,(const T*)dv,dc,ds,L.s_b,L.s_p,L.s_h,D,H,BS,L.kps,vqb,L.vdb);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<uint8_t> ref(CB,0);
    for(int t=0;t<N;++t){ if(slots[t]<0) continue;
        for(int h=0;h<H;++h){ const int64_t sb=slot_base(L,slots[t],h),base=(int64_t)(t*H+h)*D;
            for(int d=0;d<D;++d) ref[sb+d]=ref_e4b15(kf[base+d]);
            ref_value_q(&vf[base],ref.data(),sb,D,L.kps,vqb,L.vdb); } }
    auto got=d2h(dc,CB); long mm=0; for(size_t i=0;i<CB;++i) mm+=got[i]!=ref[i];
    rep_ex(nm,mm);
    CK(cudaFree(dk));CK(cudaFree(dv));CK(cudaFree(ds));CK(cudaFree(dc));
}

// ---- MSE-key store: kernel vs host replay ----
static void test_store_mse(const char* nm,int D,int mse_bits,int vqb){
    const int N=12,H=2,BS=16,NB=6,NC=1<<mse_bits;
    const int mse_bytes=D*mse_bits/8;
    Layout L=mk_layout(D,H,BS,mse_bytes+2,vqb);
    std::vector<int> slots(N); std::vector<int> perm(NB*BS); for(int i=0;i<NB*BS;++i)perm[i]=i;
    std::shuffle(perm.begin(),perm.end(),rng);
    for(int t=0;t<N;++t) slots[t]=perm[t];
    slots[7]=-1;
    auto y=rv((size_t)N*H*D,-1,1), vf=rv((size_t)N*H*D,-2,2), norms=rv((size_t)N*H,0.5f,2.0f);
    auto cents=rv(NC,-1,1); std::sort(cents.begin(),cents.end());
    std::vector<float> mids(NC-1); for(int i=0;i<NC-1;++i) mids[i]=0.5f*(cents[i]+cents[i+1]);
    auto dy=dnew(y); auto dvv=dnew(vf); auto dn=dnew(norms); auto dm=dnew(mids); auto ds=dnew(slots);
    const size_t CB=(size_t)NB*BS*H*L.slot;
    uint8_t* dc=dz<uint8_t>(CB);
    tq_store_mse<int><<<N*H,32>>>(dy,dn,dvv,dm,dc,ds,L.s_b,L.s_p,L.s_h,D,H,BS,mse_bits,mse_bytes,NC,L.kps,vqb,L.vdb);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<uint8_t> ref(CB,0);
    for(int t=0;t<N;++t){ if(slots[t]<0) continue;
        for(int h=0;h<H;++h){ const int64_t sb=slot_base(L,slots[t],h),base=(int64_t)(t*H+h)*D;
            std::vector<uint8_t> idx(D);
            for(int d=0;d<D;++d){ const float yv=y[base+d]; int lo=0,hi=NC-1;
                for(int it=0;it<mse_bits;++it){int mid=(lo+hi)>>1; if(yv>=mids[std::min(mid,NC-2)])lo=mid+1;else hi=mid;}
                idx[d]=uint8_t(std::min(lo,NC-1)); }
            if(mse_bits==4) for(int j=0;j<mse_bytes;++j) ref[sb+j]=uint8_t((idx[2*j]&0xF)|((idx[2*j+1]&0xF)<<4));
            else for(int g=0;g<D/8;++g){uint32_t p=0;for(int i=0;i<8;++i)p|=uint32_t(idx[8*g+i]&0x7)<<(3*i);
                ref[sb+3*g]=uint8_t(p&0xFF);ref[sb+3*g+1]=uint8_t((p>>8)&0xFF);ref[sb+3*g+2]=uint8_t((p>>16)&0xFF);}
            put16(&ref[sb+mse_bytes],f2h_rn(norms[t*H+h]));
            ref_value_q(&vf[base],ref.data(),sb,D,L.kps,vqb,L.vdb); } }
    auto got=d2h(dc,CB); long mm=0; for(size_t i=0;i<CB;++i) mm+=got[i]!=ref[i];
    rep_ex(nm,mm);
    CK(cudaFree(dy));CK(cudaFree(dvv));CK(cudaFree(dn));CK(cudaFree(dm));CK(cudaFree(ds));CK(cudaFree(dc));
}

// ---- decode fixture: cache filled through the store kernel ----
struct Fixture{
    int B,Hk,Hq,D,BS,NBLK; std::vector<int> seq,bt; Layout L;
    std::vector<uint8_t> cache; std::vector<float> cents;
    int mse_bits,mse_bytes,key_fp8,nc;
    uint8_t* dc; int* dbt; int* dseq; float* dcent;
};
static Fixture mk_fixture(bool fp8,int D,int vqb,int mse_bits,int nc){
    Fixture F; F.B=2;F.Hk=2;F.Hq=4;F.D=D;F.BS=16;F.seq={37,53};
    F.mse_bits=fp8?0:mse_bits; F.mse_bytes=fp8?0:D*mse_bits/8; F.key_fp8=fp8; F.nc=nc;
    F.L=mk_layout(D,F.Hk,F.BS,fp8?D:F.mse_bytes+2,vqb);
    const int bpb=(53+F.BS-1)/F.BS; F.NBLK=F.B*bpb;
    F.bt.resize(F.B*bpb); std::vector<int> perm(F.NBLK); for(int i=0;i<F.NBLK;++i)perm[i]=i;
    std::shuffle(perm.begin(),perm.end(),rng); F.bt=perm;
    int N=0; for(int s:F.seq)N+=s;
    std::vector<int> slots(N); int i=0;
    for(int b=0;b<F.B;++b) for(int p=0;p<F.seq[b];++p,++i)
        slots[i]=F.bt[b*bpb+p/F.BS]*F.BS+p%F.BS;
    const int NC=1<<mse_bits;
    F.cents=rv(fp8?4:NC,-1,1); std::sort(F.cents.begin(),F.cents.end());
    const size_t CB=(size_t)F.NBLK*F.BS*F.Hk*F.L.slot;
    F.dc=dz<uint8_t>(CB);
    auto ds=dnew(slots);
    if(fp8){
        auto kf=rv((size_t)N*F.Hk*D,-2,2), vf=rv((size_t)N*F.Hk*D,-2,2);
        std::vector<uint16_t> kb(kf.size()),vb(vf.size());
        for(size_t j=0;j<kf.size();++j){kb[j]=f2h_rn(kf[j]);vb[j]=f2h_rn(vf[j]);}
        auto dk=dnew(kb); auto dv=dnew(vb);
        tq_store_fp8<__half,int><<<N*F.Hk,32>>>((const __half*)dk,(const __half*)dv,F.dc,ds,
            F.L.s_b,F.L.s_p,F.L.s_h,D,F.Hk,F.BS,F.L.kps,vqb,F.L.vdb);
        CK(cudaDeviceSynchronize());CK(cudaGetLastError());CK(cudaFree(dk));CK(cudaFree(dv));
    } else {
        auto y=rv((size_t)N*F.Hk*D,-1,1), vf=rv((size_t)N*F.Hk*D,-2,2), norms=rv((size_t)N*F.Hk,0.5f,2.0f);
        std::vector<float> mids(NC-1); for(int c=0;c<NC-1;++c)mids[c]=0.5f*(F.cents[c]+F.cents[c+1]);
        auto dy=dnew(y);auto dvv=dnew(vf);auto dn=dnew(norms);auto dm=dnew(mids);
        tq_store_mse<int><<<N*F.Hk,32>>>(dy,dn,dvv,dm,F.dc,ds,F.L.s_b,F.L.s_p,F.L.s_h,
            D,F.Hk,F.BS,mse_bits,F.mse_bytes,NC,F.L.kps,vqb,F.L.vdb);
        CK(cudaDeviceSynchronize());CK(cudaGetLastError());
        CK(cudaFree(dy));CK(cudaFree(dvv));CK(cudaFree(dn));CK(cudaFree(dm));
    }
    CK(cudaFree(ds));
    F.cache=d2h(F.dc,CB);
    F.dbt=dnew(F.bt); F.dseq=dnew(F.seq); F.dcent=dnew(F.cents);
    return F;
}
static void free_fixture(Fixture&F){CK(cudaFree(F.dc));CK(cudaFree(F.dbt));CK(cudaFree(F.dseq));CK(cudaFree(F.dcent));}

// dequantize one (b,pos,kv_head) key/value from cache bytes, in double
static void ref_kv(const Fixture&F,int b,int pos,int kh,std::vector<double>&k,std::vector<double>&v){
    const int bpb=(53+F.BS-1)/F.BS;
    const int64_t sb=int64_t(F.bt[b*bpb+pos/F.BS])*F.L.s_b+int64_t(pos%F.BS)*F.L.s_p+int64_t(kh)*F.L.s_h;
    const uint8_t* c=F.cache.data();
    if(F.key_fp8) for(int d=0;d<F.D;++d) k[d]=h2f(e4b15_f16(c[sb+d]));
    else {
        double nsq=0; for(int d=0;d<F.D;++d){k[d]=F.cents[ref_mse_idx(c,sb,F.mse_bits,d)];nsq+=k[d]*k[d];}
        const double vn=h2f(get16(c+sb+F.mse_bytes));
        const double inv=F.nc?1.0/std::sqrt(nsq+1e-16):1.0;
        for(int d=0;d<F.D;++d)k[d]*=vn*inv;
    }
    const int64_t vb=sb+F.L.kps;
    const double vs=h2f(get16(c+vb+F.L.vdb)),vz=h2f(get16(c+vb+F.L.vdb+2));
    for(int d=0;d<F.D;++d)v[d]=ref_value_idx(c,vb,F.L.vqb,d)*vs+vz;
}

// fp64 replay of stage1 spans + stage2 merge
static void ref_decode(const Fixture&F,const std::vector<float>&q,int splits,int window,
                       std::vector<double>&out,std::vector<double>&lse){
    const float scale=1.0f/std::sqrt(float(F.D));
    out.assign((size_t)F.B*F.Hq*F.D,0); lse.assign((size_t)F.B*F.Hq,0);
    std::vector<double> k(F.D),v(F.D);
    for(int b=0;b<F.B;++b) for(int h=0;h<F.Hq;++h){
        const int kh=h/(F.Hq/F.Hk), seq=F.seq[b];
        const int kv_start=window>0?std::max(seq-window,0):0;
        const int split_len=(seq-kv_start+splits-1)/splits;
        const int kps2=(seq+splits-1)/splits;
        std::vector<double> sm(splits),sl(splits),so((size_t)splits*F.D);
        std::vector<bool> live(splits,false);
        for(int s=0;s<splits;++s){
            const int ss=kv_start+split_len*s, se=std::min(ss+split_len,seq);
            const bool st2=std::min(kps2*s+kps2,seq)>kps2*s;
            if(ss>=se){ if(st2){printf("split-set mismatch b=%d s=%d\n",b,s);++g_fail;} continue; }
            if(!st2){printf("split-set mismatch b=%d s=%d\n",b,s);++g_fail;continue;}
            live[s]=true;
            double m=-INFINITY;
            std::vector<double> raw(se-ss);
            for(int kv=ss;kv<se;++kv){ ref_kv(F,b,kv,kh,k,v);
                double dot=0; for(int d=0;d<F.D;++d)dot+=double(q[((size_t)b*F.Hq+h)*F.D+d])*k[d];
                raw[kv-ss]=dot*scale; m=std::max(m,raw[kv-ss]); }
            double l=0; std::vector<double> o(F.D,0);
            for(int kv=ss;kv<se;++kv){ const double p=std::exp(raw[kv-ss]-m); l+=p;
                ref_kv(F,b,kv,kh,k,v); for(int d=0;d<F.D;++d)o[d]+=p*v[d]; }
            sm[s]=m; sl[s]=l; for(int d=0;d<F.D;++d)so[(size_t)s*F.D+d]=o[d]/l;
        }
        double em=-INFINITY,es=0; std::vector<double> acc(F.D,0);
        for(int s=0;s<splits;++s){ if(!live[s])continue;
            const double t=sm[s]+std::log(sl[s]), nm=std::max(t,em);
            const double os=std::exp(em-nm), el=std::exp(t-nm);
            for(int d=0;d<F.D;++d)acc[d]=acc[d]*os+el*so[(size_t)s*F.D+d];
            es=es*os+el; em=nm; }
        for(int d=0;d<F.D;++d)out[((size_t)b*F.Hq+h)*F.D+d]=acc[d]/es;
        lse[(size_t)b*F.Hq+h]=em+std::log(es);
    }
}

template <typename TQ>
static void test_decode(const char* nm,Fixture&F,const std::vector<float>&qf,
                        const TQ* dq,int splits,int window){
    float* dmid=dz<float>((size_t)F.B*F.Hq*splits*(F.D+1));
    uint16_t* dout=dz<uint16_t>((size_t)F.B*F.Hq*F.D);
    float* dlse=dz<float>((size_t)F.B*F.Hq);
    const int bpb=(53+F.BS-1)/F.BS;
    const int64_t sms=F.D+1,smh=(int64_t)splits*sms,smb=(int64_t)F.Hq*smh;
    dim3 g1(F.B,F.Hq,splits);
    tq_decode_stage1<TQ><<<g1,32>>>((const TQ*)dq,F.dc,F.dbt,F.dseq,F.dcent,dmid,
        (int64_t)F.Hq*F.D,F.D,F.L.s_b,F.L.s_p,F.L.s_h,bpb,smb,smh,sms,
        F.D,F.BS,splits,F.Hq/F.Hk,F.mse_bits,F.mse_bytes,F.L.kps,F.L.vqb,F.L.vdb,
        1.0f/std::sqrt(float(F.D)),F.key_fp8,F.nc,window);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    dim3 g2(F.B,F.Hq);
    tq_decode_stage2<__half><<<g2,32>>>(dmid,(__half*)dout,dlse,F.dseq,smb,smh,sms,
        (int64_t)F.Hq*F.D,F.D,F.Hq,splits,F.D);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<double> rout,rlse; ref_decode(F,qf,splits,window,rout,rlse);
    auto out=d2h(dout,(size_t)F.B*F.Hq*F.D); auto lseh=d2h(dlse,(size_t)F.B*F.Hq);
    double gs=0,rs=0,ls=0,lr=0;
    for(size_t i=0;i<rout.size();++i){gs+=std::abs(h2f(out[i])-rout[i]);rs+=std::abs(rout[i]);}
    for(size_t i=0;i<rlse.size();++i){ls+=std::abs(lseh[i]-rlse[i]);lr+=std::abs(rlse[i]);}
    char buf[96]; snprintf(buf,sizeof buf,"%s out",nm); rep_rel(buf,gs/std::max(rs,1e-30),1e-3);
    snprintf(buf,sizeof buf,"%s lse",nm); rep_rel(buf,ls/std::max(lr,1e-30),5e-5);
    CK(cudaFree(dmid));CK(cudaFree(dout));CK(cudaFree(dlse));
}

int main(){
    // ---- store kernels: byte-exact host replay ----
    test_store_fp8<__half>("store fp8 k8v4 D=64 f16",64,4,false);
    test_store_fp8<__half>("store fp8 k8v3 D=64 f16",64,3,false);
    test_store_fp8<__nv_bfloat16>("store fp8 k8v4 D=128 bf16",128,4,true);
    test_store_mse("store mse 4bit D=64",64,4,4);
    test_store_mse("store mse 3bit D=64",64,3,4);

    // ---- decode: fp64 softmax reference over the same cache bytes ----
    { Fixture F=mk_fixture(true,64,4,0,0);
      auto qf=rv((size_t)F.B*F.Hq*F.D,-1,1);
      std::vector<uint16_t> qh(qf.size());
      for(size_t i=0;i<qf.size();++i){qh[i]=f2h_rn(qf[i]);qf[i]=h2f(qh[i]);}
      auto dqh=dnew(qh);
      test_decode<__half>("decode k8v4 splits=1",F,qf,(const __half*)dqh,1,0);
      test_decode<__half>("decode k8v4 splits=5",F,qf,(const __half*)dqh,5,0);
      test_decode<__half>("decode k8v4 win=24 splits=3",F,qf,(const __half*)dqh,3,24);
      CK(cudaFree(dqh));

      // ---- full dequant, fp8 path: bit-exact host replay ----
      { const int b=0,S=F.seq[0],bpb=(53+F.BS-1)/F.BS;
        uint16_t* dko=dz<uint16_t>((size_t)F.Hk*S*F.D); uint16_t* dvo=dz<uint16_t>((size_t)F.Hk*S*F.D);
        dim3 g(S,F.Hk);
        tq_full_dequant_kv<<<g,32>>>(F.dc,F.dbt+b*bpb,F.dcent,(__half*)dko,(__half*)dvo,
            0,(int64_t)S*F.D,F.D, 0,(int64_t)S*F.D,F.D, F.L.s_b,F.L.s_p,F.L.s_h,bpb,
            F.D,F.BS,F.Hk,F.mse_bytes,F.L.kps,F.L.vqb,F.L.vdb,F.mse_bits,1,0);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        auto ko=d2h(dko,(size_t)F.Hk*S*F.D); auto vo=d2h(dvo,(size_t)F.Hk*S*F.D);
        long mm=0;
        for(int h=0;h<F.Hk;++h) for(int p=0;p<S;++p){
            const int64_t sb=int64_t(F.bt[b*bpb+p/F.BS])*F.L.s_b+int64_t(p%F.BS)*F.L.s_p+int64_t(h)*F.L.s_h;
            const int64_t vb=sb+F.L.kps;
            const float vs=h2f(get16(&F.cache[vb+F.L.vdb])),vz=h2f(get16(&F.cache[vb+F.L.vdb+2]));
            for(int d=0;d<F.D;++d){
                mm+=ko[((size_t)h*S+p)*F.D+d]!=e4b15_f16(F.cache[sb+d]);
                mm+=vo[((size_t)h*S+p)*F.D+d]!=f2h_rn(std::fmaf(ref_value_idx(F.cache.data(),vb,F.L.vqb,d),vs,vz));
            } }
        rep_ex("dequant k8v4 D=64",mm);
        CK(cudaFree(dko));CK(cudaFree(dvo)); }
      free_fixture(F); }

    { Fixture F=mk_fixture(false,64,4,4,1);
      auto qf=rv((size_t)F.B*F.Hq*F.D,-1,1);
      auto dqf=dnew(qf);
      test_decode<float>("decode mse4nc splits=2",F,qf,dqf,2,0);
      CK(cudaFree(dqf));
      free_fixture(F); }

    printf("\n%s (%d failures)\n",g_fail?"FAILED":"ALL PASS",g_fail);
    return g_fail?1:0;
}
