// csrc/fused_experts.cu
//
// MoE experts forward for the transformers experts layout, written once for CUDA and ROCm (hipified by
// kernel-builder). Weights are K-contiguous:
//   gate_up_proj [E, 2I, H]   rows [0, I) gate, [I, 2I) up
//   down_proj    [E, H, I]
//
//   out[t] = sum_k w[t, k] * down[e](act(gate[e] x[t]) * up[e] x[t]),   e = top_k_index[t, k]
//   act = SiLU (Qwen, Kimi, OLMoE, ...) or GELU tanh (Gemma 4)
//
// Every path runs the same four steps, with no host sync and static launch sizes (CUDA-graph capturable):
//   1. align:   bucket (token, expert) pairs per expert into fixed-row blocks (one CTA, device only)
//   2. gate_up: act[p, :] = silu(x[t] @ Wg[e]^T) * (x[t] @ Wu[e]^T)
//   3. down:    y[p, :]   = w[p] * act[p] @ Wd[e]^T
//   4. reduce:  out[t, :] = sum_k y[t*K + k, :]   (fp32, deterministic)
//
// The path is picked from the average rows per expert (S / E, S = T * K pairs):
//   * decode (S / E < tile_min_rows): each active expert sees a handful of rows, so the work streams the
//     active experts' weights from HBM. Lane groups do 16-byte GEMV loads with up to 8 activation rows
//     staged in shared memory; blocks with >= mfma_min rows use a 16x16 matrix-core tile instead.
//   * prefill: grouped GEMM over 16-row (BK 64) or 64-row (BK 32) blocks per expert, operands staged in
//     shared memory with register prefetch.
// Matrix cores: MFMA 16x16x16 on ROCm, mma.sync m16n8k16 on CUDA sm80+ (older GPUs use the GEMV path only).

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#define YAMOE_AMD 1
constexpr int kWarp = 64;
#define SHFL_XOR(v, o) __shfl_xor((v), (o))
#define SHFL_UP(v, o) __shfl_up((v), (o))
#else
constexpr int kWarp = 32;
#define SHFL_XOR(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
#define SHFL_UP(v, o) __shfl_up_sync(0xffffffffu, (v), (o))
#endif

constexpr int kMaxM = 8;          // max token rows per block (runtime `bm` is a power of two <= kMaxM)
constexpr int kSmemBudget = 48 * 1024;  // staging buffer budget, no opt-in needed on CUDA
constexpr int kMaxThreads = 1024;  // max threads per CTA
constexpr int kMaxExperts = 1024;  // align kernel scans one expert per thread

namespace {

// Sum over an aligned group of LANES consecutive lanes (LANES <= warp size, power of two).
template <int LANES>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int o = LANES / 2; o > 0; o >>= 1) v += SHFL_XOR(v, o);
  return v;
}

template <typename scalar_t>
__device__ __forceinline__ void load8(const scalar_t* p, float out[8]) {
  uint4 raw = *reinterpret_cast<const uint4*>(p);
  const scalar_t* s = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = static_cast<float>(s[i]);
}

__device__ __forceinline__ float dot8(const float a[8], const float b[8]) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s = fmaf(a[i], b[i], s);
  return s;
}

// Gate activation, picked at runtime (uniform branch): 0 = SiLU, 1 = GELU with the tanh approximation.
constexpr int kActSilu = 0;
constexpr int kActGeluTanh = 1;
__device__ __forceinline__ float gate_act(float g, int act) {
  if (act == kActGeluTanh) return 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
  return g / (1.f + __expf(-g));
}

// ---------------------------------------------------------------------------------------------
// Matrix-core fragments, one 16 x 16 output tile per wavefront / warp and 32 k per step.
//
// The same kernels run on both backends through four operations:
//   load_a / load_b: 16-byte loads of A rows / B columns (W rows) at k offsets owned by this lane
//   frag_mma:        accumulate the 16 x 16 x 32 product
//   frag_coord:      (row, col) of accumulator element q for this lane
//
// AMD (wave64, mfma_f32_16x16x16): lane l, r = l % 16, g = l / 16 loads row/col r at k + 8g. Two MFMAs per
//   step map logical k (4g + j) to physical k + 8g + j and k + 8g + 4 + j. Acc element q is (4g + q, r).
// CUDA (warp32, mma.m16n8k16, sm80+): lane l, g = l / 4, t = l % 4 loads rows/cols g and g + 8 at k + 8t.
//   The thread's logical k slots {2t, 2t+1, 2t+8, 2t+9} map to physical k + 8t + {0..3} (first mma) and
//   k + 8t + {4..7} (second). Two n8 halves make the 16 columns. Acc element q = 4 * nh + qq is
//   (g + 8 * (qq / 2), 8 * nh + 2t + qq % 2).
// A and B always use the same k mapping, so the reduction is exact. Rows past the expert's count load zeros.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }

// ---------------------------------------------------------------------------------------------
// Expert weight access. Weights are either the activation dtype (bf16 / f16) or FP8 e4m3fn with
// per-block fp32 scales [E, ceil(N / bn), ceil(K / bk)] (the transformers / DeepSeek fine-grained
// layout). FP8 is dequantized on load (weight-only, activations stay bf16 / f16):
//   e4m3fn bits s.eeee.mmm placed at fp32 bits 31 / 26..20 give 2^(e-127) * 1.m, so * 2^120 restores
//   2^(e-7) * 1.m; subnormals (e = 0) map to fp32 subnormals and come out exact the same way.
// ---------------------------------------------------------------------------------------------
struct WSrc {
  const void* w;   // [E, N, K]
  const float* s;  // FP8 block scales, else nullptr
  int sN, sK, bn, bk;
};

// Four e4m3fn bytes -> four floats equal to value / 256. Two bytes at a time become an fp16 pair with
// three integer ops (sign to bit 15, exponent and mantissa shifted into fp16 position), then the hardware
// fp16 -> fp32 conversion handles normals and subnormals exactly; fp16's exponent bias is 8 larger.
__device__ __forceinline__ void fp8x4_div256(unsigned w, float* out) {
  const unsigned a = ((w & 0x00800080u) << 8) | ((w & 0x007F007Fu) << 7);  // bytes 0, 2
  const unsigned b = (w & 0x80008000u) | ((w & 0x7F007F00u) >> 1);         // bytes 1, 3
  const float2 fa = __half22float2(*reinterpret_cast<const __half2*>(&a));
  const float2 fb = __half22float2(*reinterpret_cast<const __half2*>(&b));
  out[0] = fa.x, out[1] = fb.x, out[2] = fa.y, out[3] = fb.y;
}

__device__ __forceinline__ void fp8x8_to_float(uint2 raw, float scale, float out[8]) {
  fp8x4_div256(raw.x, out);
  fp8x4_div256(raw.y, out + 4);
  const float sc = scale * 256.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] *= sc;
}

template <typename scalar_t>
__device__ __forceinline__ uint4 pack8(const float f[8]) {
  uint4 r;
  scalar_t* o = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = static_cast<scalar_t>(f[i]);
  return r;
}

// One expert's [N, K] weight matrix.
template <typename scalar_t, bool FP8>
struct WExp {
  const unsigned char* w;
  const float* s;
  int K, sK, bn, bk;
  __device__ __forceinline__ float scale(int n, int k) const { return s[(n / bn) * sK + k / bk]; }
  // 8 consecutive weights of row n starting at k, as fp32
  __device__ __forceinline__ void load8f(int n, int k, float out[8]) const {
    if constexpr (FP8)
      fp8x8_to_float(*reinterpret_cast<const uint2*>(w + (int64_t)n * K + k), scale(n, k), out);
    else
      load8(reinterpret_cast<const scalar_t*>(w) + (int64_t)n * K + k, out);
  }
  // FP8 only: 16 consecutive weights of row n from k (16-byte load) as value / 256, and the block scale * 256.
  // 16 | block_k, so the chunk shares one scale.
  __device__ __forceinline__ float load16u(int n, int k, float out[16]) const {
    const uint4 raw = *reinterpret_cast<const uint4*>(w + (int64_t)n * K + k);
    fp8x4_div256(raw.x, out);
    fp8x4_div256(raw.y, out + 4);
    fp8x4_div256(raw.z, out + 8);
    fp8x4_div256(raw.w, out + 12);
    return scale(n, k) * 256.f;
  }
  // FP8 only: raw 8 bytes and the block scale, for loaders that dequantize later (latency hiding)
  __device__ __forceinline__ uint2 raw8(int n, int k) const { return *reinterpret_cast<const uint2*>(w + (int64_t)n * K + k); }
  // the same 8 weights packed as scalar_t (matrix-core operand)
  __device__ __forceinline__ uint4 load8p(int n, int k) const {
    if constexpr (FP8) {
      float f[8];
      load8f(n, k, f);
      return pack8<scalar_t>(f);
    } else {
      return ld16(reinterpret_cast<const scalar_t*>(w) + (int64_t)n * K + k);
    }
  }
};

template <typename scalar_t, bool FP8>
__device__ __forceinline__ WExp<scalar_t, FP8> expert_w(const WSrc& src, int e, int N, int K) {
  WExp<scalar_t, FP8> r;
  const int64_t esz = FP8 ? 1 : sizeof(scalar_t);
  r.w = reinterpret_cast<const unsigned char*>(src.w) + (int64_t)e * N * K * esz;
  r.s = FP8 ? src.s + (int64_t)e * src.sN * src.sK : nullptr;
  r.K = K, r.sK = src.sK, r.bn = src.bn, r.bk = src.bk;
  return r;
}

#ifdef YAMOE_AMD
constexpr int kAcc = 4;
struct FragA { uint4 v; };
struct FragB { uint4 v; };

typedef short yamoe_s16x4 __attribute__((ext_vector_type(4)));
typedef _Float16 yamoe_h16x4 __attribute__((ext_vector_type(4)));
typedef float yamoe_f32x4 __attribute__((ext_vector_type(4)));

template <typename scalar_t>
__device__ __forceinline__ yamoe_f32x4 mfma16(uint2 a, uint2 b, yamoe_f32x4 c);
template <>
__device__ __forceinline__ yamoe_f32x4 mfma16<at::BFloat16>(uint2 a, uint2 b, yamoe_f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(__builtin_bit_cast(yamoe_s16x4, a),
                                                   __builtin_bit_cast(yamoe_s16x4, b), c, 0, 0, 0);
}
template <>
__device__ __forceinline__ yamoe_f32x4 mfma16<at::Half>(uint2 a, uint2 b, yamoe_f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_bit_cast(yamoe_h16x4, a),
                                               __builtin_bit_cast(yamoe_h16x4, b), c, 0, 0, 0);
}

template <typename F>
__device__ __forceinline__ FragA load_a(F&& rowk, int k) {
  const int lane = threadIdx.x % kWarp;
  return FragA{rowk(lane % 16, k + 8 * (lane / 16))};
}
template <typename F>
__device__ __forceinline__ FragB load_b(F&& colk, int k) {
  const int lane = threadIdx.x % kWarp;
  return FragB{colk(lane % 16, k + 8 * (lane / 16))};
}
template <typename scalar_t>
__device__ __forceinline__ void frag_mma(float (&c)[kAcc], const FragA& a, const FragB& b) {
  yamoe_f32x4 acc = {c[0], c[1], c[2], c[3]};
  acc = mfma16<scalar_t>(make_uint2(a.v.x, a.v.y), make_uint2(b.v.x, b.v.y), acc);
  acc = mfma16<scalar_t>(make_uint2(a.v.z, a.v.w), make_uint2(b.v.z, b.v.w), acc);
  c[0] = acc[0], c[1] = acc[1], c[2] = acc[2], c[3] = acc[3];
}
__device__ __forceinline__ void frag_coord(int q, int& row, int& col) {
  const int lane = threadIdx.x % kWarp;
  row = 4 * (lane / 16) + q;
  col = lane % 16;
}
#else
constexpr int kAcc = 8;
struct FragA { uint4 lo, hi; };  // rows g, g + 8
struct FragB { uint4 lo, hi; };  // cols g, g + 8

template <typename scalar_t>
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
#else
  __trap();  // the host never routes work to the mma paths below sm80
#endif
}

template <typename F>
__device__ __forceinline__ FragA load_a(F&& rowk, int k) {
  const int lane = threadIdx.x % kWarp, g = lane / 4, t = lane % 4;
  return FragA{rowk(g, k + 8 * t), rowk(g + 8, k + 8 * t)};
}
template <typename F>
__device__ __forceinline__ FragB load_b(F&& colk, int k) {
  const int lane = threadIdx.x % kWarp, g = lane / 4, t = lane % 4;
  return FragB{colk(g, k + 8 * t), colk(g + 8, k + 8 * t)};
}
template <typename scalar_t>
__device__ __forceinline__ void frag_mma(float (&c)[kAcc], const FragA& a, const FragB& b) {
  mma16816<scalar_t>(c + 0, a.lo.x, a.hi.x, a.lo.y, a.hi.y, b.lo.x, b.lo.y);
  mma16816<scalar_t>(c + 0, a.lo.z, a.hi.z, a.lo.w, a.hi.w, b.lo.z, b.lo.w);
  mma16816<scalar_t>(c + 4, a.lo.x, a.hi.x, a.lo.y, a.hi.y, b.hi.x, b.hi.y);
  mma16816<scalar_t>(c + 4, a.lo.z, a.hi.z, a.lo.w, a.hi.w, b.hi.z, b.hi.w);
}
__device__ __forceinline__ void frag_coord(int q, int& row, int& col) {
  const int lane = threadIdx.x % kWarp, g = lane / 4, t = lane % 4;
  const int nh = q / 4, qq = q % 4;
  row = g + 8 * (qq / 2);
  col = 8 * nh + 2 * t + (qq % 2);
}
#endif

// ---------------------------------------------------------------------------------------------
// Decode: matrix-core tile for blocks with several rows. Each wavefront / warp owns 16-column tiles,
// activations come from the shared-memory staging buffer, weights straight from global memory.
// Requires the reduction dim % 32 == 0.
// ---------------------------------------------------------------------------------------------
template <typename scalar_t, bool FP8>
__device__ __forceinline__ void gate_up_mma(const scalar_t* xs, const WExp<scalar_t, FP8>& wgu,
                                            scalar_t* __restrict__ act, const int* pairs, int m, int H, int I,
                                            int c0, int c1, int act_fn) {
  const int wave = threadIdx.x / kWarp, nwaves = blockDim.x / kWarp;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  auto lda = [&](int row, int k) { return row < m ? ld16(xs + row * H + k) : zero; };
  for (int cb = c0 + wave * 16; cb < c1; cb += nwaves * 16) {
    auto ldg = [&](int col, int k) { return wgu.load8p(min(cb + col, c1 - 1), k); };
    auto ldu = [&](int col, int k) { return wgu.load8p(I + min(cb + col, c1 - 1), k); };
    float ag[kAcc] = {}, au[kAcc] = {};
#pragma unroll 2
    for (int k = 0; k < H; k += 32) {
      const FragA a = load_a(lda, k);
      const FragB bg = load_b(ldg, k), bu = load_b(ldu, k);
      frag_mma<scalar_t>(ag, a, bg);
      frag_mma<scalar_t>(au, a, bu);
    }
#pragma unroll
    for (int q = 0; q < kAcc; ++q) {
      int row, col;
      frag_coord(q, row, col);
      if (row < m && cb + col < c1) {
        const float gv = ag[q], uv = au[q];
        act[(int64_t)pairs[row] * I + cb + col] = static_cast<scalar_t>(gate_act(gv, act_fn) * uv);
      }
    }
  }
}

template <typename scalar_t, bool FP8>
__device__ __forceinline__ void down_mma(const scalar_t* as, const WExp<scalar_t, FP8>& wde,
                                         const scalar_t* __restrict__ rw, scalar_t* __restrict__ y, const int* pairs,
                                         int m, int H, int I, int c0, int c1) {
  const int wave = threadIdx.x / kWarp, nwaves = blockDim.x / kWarp;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  auto lda = [&](int row, int k) { return row < m ? ld16(as + row * I + k) : zero; };
  for (int cb = c0 + wave * 16; cb < c1; cb += nwaves * 16) {
    auto ldw = [&](int col, int k) { return wde.load8p(min(cb + col, c1 - 1), k); };
    float acc[kAcc] = {};
#pragma unroll 2
    for (int k = 0; k < I; k += 32) frag_mma<scalar_t>(acc, load_a(lda, k), load_b(ldw, k));
#pragma unroll
    for (int q = 0; q < kAcc; ++q) {
      int row, col;
      frag_coord(q, row, col);
      if (row < m && cb + col < c1) {
        const int p = pairs[row];
        y[(int64_t)p * H + cb + col] = static_cast<scalar_t>(acc[q] * static_cast<float>(rw[p]));
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Prefill: grouped GEMM over TM-row blocks of one expert x tile_n<TM>() output columns, 256 threads.
//   GATED  (gate/up): A row = x[pair / topk], W rows c (gate) and N + c (up), out = act[pair, c] = silu(g) * u
//   !GATED (down):    A row = act[pair],      W row c,                      out = y[pair, c]   = acc * rw[pair]
// Per BK-wide k-step the CTA stages the A tile and W tile(s) in shared memory (16-byte chunks spread over
// the threads) and prefetches the next step into registers while the matrix cores work.
// Waves / warps are arranged (TM / WM) x (TN / WN) with WM = min(TM, 32).
// ---------------------------------------------------------------------------------------------
constexpr int kTileM = 64;
constexpr int kTileThreads = 256;
constexpr int kTileWaves = kTileThreads / kWarp;

template <int TM>
__host__ __device__ constexpr int tile_wm() { return TM < 32 ? TM : 32; }
template <int TM>
__host__ __device__ constexpr int tile_wn() {
#ifdef YAMOE_AMD
  return TM >= 64 ? 32 : 16;
#else
  return 16;
#endif
}
template <int TM>
__host__ __device__ constexpr int tile_n() { return (kTileWaves / (TM / tile_wm<TM>())) * tile_wn<TM>(); }

template <typename scalar_t, bool FP8, bool GATED, int TM, int BK>
__global__ void __launch_bounds__(kTileThreads) moe_tile_kernel(
    const scalar_t* __restrict__ a_src, const WSrc w, scalar_t* __restrict__ out,
    const scalar_t* __restrict__ rw, const int* __restrict__ perm, const int* __restrict__ blk_expert,
    const int* __restrict__ blk_row, const int* __restrict__ blk_end, int N, int Kd, int adiv, int E, int act_fn) {
  constexpr int TN = tile_n<TM>();
  constexpr int WM = tile_wm<TM>(), WN = tile_wn<TM>();
  constexpr int FI = WM / 16, FJ = WN / 16;
  constexpr int WAVES_N = TN / WN;
  constexpr int kLds = BK + 8;  // padded shared-memory row (elements), keeps 16-byte alignment
  constexpr int CPR = BK / 8;   // 16-byte chunks per tile row
  constexpr int NA = (TM * CPR + kTileThreads - 1) / kTileThreads;
  constexpr int NB = (TN * CPR + kTileThreads - 1) / kTileThreads;
  static_assert((TM / WM) * WAVES_N == kTileWaves, "wave grid must use all waves");
  __shared__ __align__(16) scalar_t As[TM * kLds];
  __shared__ __align__(16) scalar_t Bg[TN * kLds];
  __shared__ __align__(16) scalar_t Bu[GATED ? TN * kLds : 8];
  __shared__ int pairs[TM];
  const int b = blockIdx.x;
  const int e = blk_expert[b];
  if (e >= E) return;
  const int row0 = blk_row[b];
  const int m = min(blk_end[b] - row0, TM);
  const int tid = threadIdx.x;
  if (tid < TM) pairs[tid] = tid < m ? perm[row0 + tid] : 0;
  __syncthreads();

  const int n_base = blockIdx.y * TN;
  const auto W = expert_w<scalar_t, FP8>(w, e, (GATED ? 2 : 1) * N, Kd);
  const uint4 zero = make_uint4(0, 0, 0, 0);
  // loader: chunk id c -> (row c / CPR, k offset (c % CPR) * 8)
  const scalar_t* a_ld[NA];
  bool a_on[NA], a_ok[NA];
  int a_off[NA];
#pragma unroll
  for (int i = 0; i < NA; ++i) {
    const int c = tid + i * kTileThreads, row = c / CPR;
    a_on[i] = c < TM * CPR;
    a_ok[i] = a_on[i] && row < m;
    a_off[i] = row * kLds + (c % CPR) * 8;
    a_ld[i] = a_src + (int64_t)(pairs[a_ok[i] ? row : 0] / adiv) * Kd + (c % CPR) * 8;
  }
  int b_off[NB], b_row[NB], b_k[NB];
#pragma unroll
  for (int i = 0; i < NB; ++i) {
    const int c = tid + i * kTileThreads, row = c / CPR;
    b_off[i] = row * kLds + (c % CPR) * 8;
    b_row[i] = min(n_base + row, N - 1);
    b_k[i] = (c % CPR) * 8;
  }
  uint4 ra[NA], rg[NB], ru[NB];
  // FP8: prefetch raw bytes + scales, dequantize when storing to shared memory
  uint2 qg[NB], qu[NB];
  float sg[NB], su[NB];
  auto fetch_b = [&](int k) {
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      if constexpr (FP8) {
        qg[i] = W.raw8(b_row[i], b_k[i] + k);
        sg[i] = W.scale(b_row[i], b_k[i] + k);
        if (GATED) qu[i] = W.raw8(N + b_row[i], b_k[i] + k), su[i] = W.scale(N + b_row[i], b_k[i] + k);
      } else {
        rg[i] = W.load8p(b_row[i], b_k[i] + k);
        if (GATED) ru[i] = W.load8p(N + b_row[i], b_k[i] + k);
      }
    }
  };
  auto b_packed = [&](int i, bool up) -> uint4 {
    if constexpr (FP8) {
      float f[8];
      fp8x8_to_float(up ? qu[i] : qg[i], up ? su[i] : sg[i], f);
      return pack8<scalar_t>(f);
    } else {
      return up ? ru[i] : rg[i];
    }
  };
#pragma unroll
  for (int i = 0; i < NA; ++i) ra[i] = a_ok[i] ? ld16(a_ld[i]) : zero;
  fetch_b(0);

  const int wave = tid / kWarp;
  const int wr = (wave / WAVES_N) * WM, wc = (wave % WAVES_N) * WN;
  float ag[FI][FJ][kAcc] = {}, au[FI][FJ][kAcc] = {};

  for (int k = 0; k < Kd; k += BK) {
#pragma unroll
    for (int i = 0; i < NA; ++i)
      if (a_on[i]) *reinterpret_cast<uint4*>(&As[a_off[i]]) = ra[i];
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      *reinterpret_cast<uint4*>(&Bg[b_off[i]]) = b_packed(i, false);
      if (GATED) *reinterpret_cast<uint4*>(&Bu[b_off[i]]) = b_packed(i, true);
    }
    __syncthreads();
    if (k + BK < Kd) {
#pragma unroll
      for (int i = 0; i < NA; ++i) ra[i] = a_ok[i] ? ld16(a_ld[i] + k + BK) : zero;
      fetch_b(k + BK);
    }
#pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      FragA fa[FI];
      FragB fg[FJ], fu[FJ];
#pragma unroll
      for (int i = 0; i < FI; ++i)
        fa[i] = load_a([&](int row, int kx) { return ld16(&As[(wr + i * 16 + row) * kLds + kx]); }, kk);
#pragma unroll
      for (int j = 0; j < FJ; ++j) {
        fg[j] = load_b([&](int col, int kx) { return ld16(&Bg[(wc + j * 16 + col) * kLds + kx]); }, kk);
        if (GATED) fu[j] = load_b([&](int col, int kx) { return ld16(&Bu[(wc + j * 16 + col) * kLds + kx]); }, kk);
      }
#pragma unroll
      for (int i = 0; i < FI; ++i)
#pragma unroll
        for (int j = 0; j < FJ; ++j) {
          frag_mma<scalar_t>(ag[i][j], fa[i], fg[j]);
          if (GATED) frag_mma<scalar_t>(au[i][j], fa[i], fu[j]);
        }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < FI; ++i)
#pragma unroll
    for (int j = 0; j < FJ; ++j)
#pragma unroll
      for (int q = 0; q < kAcc; ++q) {
        int fr, fc;
        frag_coord(q, fr, fc);
        const int row = wr + i * 16 + fr, c = n_base + wc + j * 16 + fc;
        if (row < m && c < N) {
          const int p = pairs[row];
          float v;
          if (GATED) {
            const float gv = ag[i][j][q], uv = au[i][j][q];
            v = gate_act(gv, act_fn) * uv;
          } else {
            v = ag[i][j][q] * static_cast<float>(rw[p]);
          }
          out[(int64_t)p * N + c] = static_cast<scalar_t>(v);
        }
      }
}

// ---------------------------------------------------------------------------------------------
// 1. routing: one CTA of kMaxExperts threads. Writes perm (sorted position -> pair) and a block
//    table (expert, first sorted row, end row) for every kMaxM-row block. Unused blocks get E.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kMaxExperts) align_kernel(const int64_t* __restrict__ ids,  // [S]
                                                           int S, int E, int max_blocks, int bm,
                                                           int* __restrict__ perm,
                                                           int* __restrict__ blk_expert,
                                                           int* __restrict__ blk_row,
                                                           int* __restrict__ blk_end) {
  __shared__ int cnt[kMaxExperts];
  __shared__ int cursor[kMaxExperts];
  __shared__ int start_of[kMaxExperts];  // exclusive prefix of rows per expert
  __shared__ int warp_rows[kMaxExperts / 32], warp_blks[kMaxExperts / 32];
  const int tid = threadIdx.x;
  const int lane = tid % kWarp, warp = tid / kWarp, nwarps = blockDim.x / kWarp;

  cnt[tid] = 0;
  cursor[tid] = 0;
  for (int b = tid; b < max_blocks; b += blockDim.x) blk_expert[b] = E;
  __syncthreads();
  for (int p = tid; p < S; p += blockDim.x) atomicAdd(&cnt[(int)ids[p]], 1);
  __syncthreads();

  // inclusive scan of rows and blocks per expert: warp shuffles, then one warp scans the warp totals
  const int c = tid < E ? cnt[tid] : 0;
  const int nb = (c + bm - 1) / bm;
  int sr = c, sb = nb;
#pragma unroll
  for (int o = 1; o < kWarp; o <<= 1) {
    const int r = SHFL_UP(sr, o), b2 = SHFL_UP(sb, o);
    if (lane >= o) sr += r, sb += b2;
  }
  if (lane == kWarp - 1) warp_rows[warp] = sr, warp_blks[warp] = sb;
  __syncthreads();
  if (warp == 0) {
    int wr = lane < nwarps ? warp_rows[lane] : 0, wb = lane < nwarps ? warp_blks[lane] : 0;
#pragma unroll
    for (int o = 1; o < kWarp; o <<= 1) {
      const int r = SHFL_UP(wr, o), b2 = SHFL_UP(wb, o);
      if (lane >= o) wr += r, wb += b2;
    }
    if (lane < nwarps) warp_rows[lane] = wr, warp_blks[lane] = wb;
  }
  __syncthreads();
  if (warp > 0) sr += warp_rows[warp - 1], sb += warp_blks[warp - 1];
  const int start = sr - c;  // exclusive
  const int bstart = sb - nb;
  start_of[tid] = start;
  __syncthreads();

  for (int p = tid; p < S; p += blockDim.x) {
    const int e = (int)ids[p];
    perm[start_of[e] + atomicAdd(&cursor[e], 1)] = p;
  }
  if (tid < E) {
    for (int j = 0; j < nb; ++j) {
      blk_expert[bstart + j] = tid;
      blk_row[bstart + j] = start + j * bm;
      blk_end[bstart + j] = start + c;
    }
  }
}

// Load up to kMaxM activation rows of length K (gathered by `rows`) into shared memory.
template <typename scalar_t>
__device__ __forceinline__ void stage_rows(scalar_t* xs, const scalar_t* __restrict__ src,
                                           const int* rows, int m, int bm, int K) {
  const int vecs = K / 8;
  for (int i = threadIdx.x; i < bm * vecs; i += blockDim.x) {
    const int r = i / vecs, v = i - r * vecs;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < m) val = *reinterpret_cast<const uint4*>(src + (int64_t)rows[r] * K + v * 8);
    *reinterpret_cast<uint4*>(xs + r * K + v * 8) = val;
  }
}

// Shared block prologue: resolve this CTA's expert and token rows. Returns m (0 = no work).
__device__ __forceinline__ int block_rows(const int* __restrict__ perm, const int* __restrict__ blk_expert,
                                          const int* __restrict__ blk_row, const int* __restrict__ blk_end,
                                          int E, int bm, int& e) {
  const int b = blockIdx.x;
  e = blk_expert[b];
  if (e >= E) return 0;
  return min(blk_end[b] - blk_row[b], bm);
}

// ---------------------------------------------------------------------------------------------
// 2. gate/up + silu. grid (max_blocks, ceil(I / cols)); each group of LANES lanes owns a column.
// ---------------------------------------------------------------------------------------------
template <typename scalar_t, bool FP8, int M, int LANES, int COLS>
__device__ __forceinline__ void gate_up_cols(const scalar_t* xs, const WExp<scalar_t, FP8>& wgu,
                                             scalar_t* __restrict__ act, const int* pairs, int m,
                                             int H, int I, int c0, int c1, int act_fn) {
  // Each group of LANES lanes computes COLS adjacent output columns at once, so every activation
  // vector read from shared memory is reused COLS times (matters when M > 1).
  const int group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
  const int ngroups = blockDim.x / LANES;
  const int vecs = H / 8;
  for (int cb = c0 + group * COLS; cb < c1; cb += ngroups * COLS) {
    int rgate[COLS];
#pragma unroll
    for (int j = 0; j < COLS; ++j) rgate[j] = min(cb + j, c1 - 1);
    float ag[COLS][M], au[COLS][M];
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) ag[j][r] = 0.f, au[j][r] = 0.f;
    if constexpr (FP8) {
      // 16 weights per 16-byte load, scale applied once per chunk
      for (int v = lane; v < H / 16; v += LANES) {
        float g16[COLS][16], u16[COLS][16], sg[COLS], su[COLS];
#pragma unroll
        for (int j = 0; j < COLS; ++j) {
          sg[j] = wgu.load16u(rgate[j], v * 16, g16[j]);
          su[j] = wgu.load16u(I + rgate[j], v * 16, u16[j]);
        }
#pragma unroll
        for (int r = 0; r < M; ++r) {
          float xa[8], xb[8];
          load8(xs + r * H + v * 16, xa);
          load8(xs + r * H + v * 16 + 8, xb);
#pragma unroll
          for (int j = 0; j < COLS; ++j) {
            ag[j][r] = fmaf(sg[j], dot8(g16[j], xa) + dot8(g16[j] + 8, xb), ag[j][r]);
            au[j][r] = fmaf(su[j], dot8(u16[j], xa) + dot8(u16[j] + 8, xb), au[j][r]);
          }
        }
      }
    } else {
    for (int v = lane; v < vecs; v += LANES) {
      float g8[COLS][8], u8[COLS][8];
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
        wgu.load8f(rgate[j], v * 8, g8[j]);
        wgu.load8f(I + rgate[j], v * 8, u8[j]);
      }
#pragma unroll
      for (int r = 0; r < M; ++r) {
        float x8[8];
        load8(xs + r * H + v * 8, x8);
#pragma unroll
        for (int j = 0; j < COLS; ++j) {
          ag[j][r] += dot8(g8[j], x8);
          au[j][r] += dot8(u8[j], x8);
        }
      }
    }
    }
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const float g = group_sum<LANES>(ag[j][r]);
        const float u = group_sum<LANES>(au[j][r]);
        if (lane == 0 && r < m && cb + j < c1)
          act[(int64_t)pairs[r] * I + cb + j] = static_cast<scalar_t>(gate_act(g, act_fn) * u);
      }
  }
}

template <typename scalar_t, bool FP8, int LANES, int CW>
__global__ void __launch_bounds__(kMaxThreads) gate_up_silu_kernel(
    const scalar_t* __restrict__ x, const WSrc w, scalar_t* __restrict__ act,
    const int* __restrict__ perm, const int* __restrict__ blk_expert, const int* __restrict__ blk_row,
    const int* __restrict__ blk_end, int H, int I, int topk, int E, int bm, int cols, int mfma_min,
    int act_fn) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* xs = reinterpret_cast<scalar_t*>(smem_raw);
  __shared__ int pairs[kMaxM], toks[kMaxM];
  int e;
  const int m = block_rows(perm, blk_expert, blk_row, blk_end, E, bm, e);
  if (m == 0) return;
  if (threadIdx.x < kMaxM) {
    const int p = threadIdx.x < m ? perm[blk_row[blockIdx.x] + threadIdx.x] : 0;
    pairs[threadIdx.x] = p;
    toks[threadIdx.x] = p / topk;
  }
  __syncthreads();
  stage_rows(xs, x, toks, m, bm, H);
  __syncthreads();
  const int c0 = blockIdx.y * cols, c1 = min(I, c0 + cols);
  const auto wg_e = expert_w<scalar_t, FP8>(w, e, 2 * I, H);
  if (m >= mfma_min && H % 32 == 0) {
    gate_up_mma<scalar_t, FP8>(xs, wg_e, act, pairs, m, H, I, c0, c1, act_fn);
    return;
  }
  if (m == 1)      gate_up_cols<scalar_t, FP8, 1, LANES, 1>(xs, wg_e, act, pairs, m, H, I, c0, c1, act_fn);
  else if (m == 2) gate_up_cols<scalar_t, FP8, 2, LANES, CW>(xs, wg_e, act, pairs, m, H, I, c0, c1, act_fn);
  else if (m <= 4) gate_up_cols<scalar_t, FP8, 4, LANES, CW>(xs, wg_e, act, pairs, m, H, I, c0, c1, act_fn);
  else             gate_up_cols<scalar_t, FP8, kMaxM, LANES, CW>(xs, wg_e, act, pairs, m, H, I, c0, c1, act_fn);
}

// ---------------------------------------------------------------------------------------------
// 3. down projection * routing weight. grid (max_blocks, ceil(H / cols)).
// ---------------------------------------------------------------------------------------------
template <typename scalar_t, bool FP8, int M, int LANES, int COLS>
__device__ __forceinline__ void down_cols(const scalar_t* as, const WExp<scalar_t, FP8>& wde,
                                          const scalar_t* __restrict__ rw, scalar_t* __restrict__ y,
                                          const int* pairs, int m, int H, int I, int c0, int c1) {
  const int group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
  const int ngroups = blockDim.x / LANES;
  const int vecs = I / 8;
  for (int cb = c0 + group * COLS; cb < c1; cb += ngroups * COLS) {
    int rrow[COLS];
#pragma unroll
    for (int j = 0; j < COLS; ++j) rrow[j] = min(cb + j, c1 - 1);
    float acc[COLS][M];
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) acc[j][r] = 0.f;
    if constexpr (FP8) {
      for (int v = lane; v < I / 16; v += LANES) {
        float w16[COLS][16], sw[COLS];
#pragma unroll
        for (int j = 0; j < COLS; ++j) sw[j] = wde.load16u(rrow[j], v * 16, w16[j]);
#pragma unroll
        for (int r = 0; r < M; ++r) {
          float aa[8], ab[8];
          load8(as + r * I + v * 16, aa);
          load8(as + r * I + v * 16 + 8, ab);
#pragma unroll
          for (int j = 0; j < COLS; ++j) acc[j][r] = fmaf(sw[j], dot8(w16[j], aa) + dot8(w16[j] + 8, ab), acc[j][r]);
        }
      }
    } else {
    for (int v = lane; v < vecs; v += LANES) {
      float w8[COLS][8];
#pragma unroll
      for (int j = 0; j < COLS; ++j) wde.load8f(rrow[j], v * 8, w8[j]);
#pragma unroll
      for (int r = 0; r < M; ++r) {
        float a8[8];
        load8(as + r * I + v * 8, a8);
#pragma unroll
        for (int j = 0; j < COLS; ++j) acc[j][r] += dot8(w8[j], a8);
      }
    }
    }
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const float sum = group_sum<LANES>(acc[j][r]);
        if (lane == 0 && r < m && cb + j < c1) {
          const int p = pairs[r];
          y[(int64_t)p * H + cb + j] = static_cast<scalar_t>(sum * static_cast<float>(rw[p]));
        }
      }
  }
}

template <typename scalar_t, bool FP8, int LANES, int CW>
__global__ void __launch_bounds__(kMaxThreads) down_kernel(
    const scalar_t* __restrict__ act, const WSrc w, const scalar_t* __restrict__ rw,
    scalar_t* __restrict__ y, const int* __restrict__ perm, const int* __restrict__ blk_expert,
    const int* __restrict__ blk_row, const int* __restrict__ blk_end, int H, int I, int E, int bm, int cols,
    int mfma_min) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* as = reinterpret_cast<scalar_t*>(smem_raw);
  __shared__ int pairs[kMaxM];
  int e;
  const int m = block_rows(perm, blk_expert, blk_row, blk_end, E, bm, e);
  if (m == 0) return;
  if (threadIdx.x < kMaxM) pairs[threadIdx.x] = threadIdx.x < m ? perm[blk_row[blockIdx.x] + threadIdx.x] : 0;
  __syncthreads();
  stage_rows(as, act, pairs, m, bm, I);
  __syncthreads();
  const int c0 = blockIdx.y * cols, c1 = min(H, c0 + cols);
  const auto wd_e = expert_w<scalar_t, FP8>(w, e, H, I);
  if (m >= mfma_min && I % 32 == 0) {
    down_mma<scalar_t, FP8>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
    return;
  }
  if (m == 1)      down_cols<scalar_t, FP8, 1, LANES, 1>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
  else if (m == 2) down_cols<scalar_t, FP8, 2, LANES, CW>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
  else if (m <= 4) down_cols<scalar_t, FP8, 4, LANES, CW>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
  else             down_cols<scalar_t, FP8, kMaxM, LANES, CW>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
}

// ---------------------------------------------------------------------------------------------
// 4. out[t, h] = sum_k y[t*K + k, h]
// ---------------------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void topk_sum_kernel(const scalar_t* __restrict__ y, scalar_t* __restrict__ out, int T, int K,
                                int H) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)T * H) return;
  const int64_t t = i / H, h = i - t * H;
  float s = 0.f;
  for (int k = 0; k < K; ++k) s += static_cast<float>(y[(t * K + k) * H + h]);
  out[i] = static_cast<scalar_t>(s);
}

struct Launch {
  int threads, cols, lanes, cw, mfma_min, act;
};

// Dispatch runtime (lanes, cw) onto template instances.
#define YAMOE_CASE(L, C, M) { constexpr int CWV = C; M(L); }
#define YAMOE_SWITCH(cfg, M)                                                     \
  if (cfg.lanes == 8 && cfg.cw == 1) YAMOE_CASE(8, 1, M)                         \
  else if (cfg.lanes == 8 && cfg.cw == 2) YAMOE_CASE(8, 2, M)                    \
  else if (cfg.lanes == 8) YAMOE_CASE(8, 4, M)                                   \
  else if (cfg.lanes == 16 && cfg.cw == 1) YAMOE_CASE(16, 1, M)                  \
  else if (cfg.lanes == 16 && cfg.cw == 2) YAMOE_CASE(16, 2, M)                  \
  else if (cfg.lanes == 16) YAMOE_CASE(16, 4, M)                                 \
  else if (cfg.lanes == 32 && cfg.cw == 1) YAMOE_CASE(32, 1, M)                  \
  else if (cfg.lanes == 32 && cfg.cw == 2) YAMOE_CASE(32, 2, M)                  \
  else if (cfg.lanes == 32) YAMOE_CASE(32, 4, M)                                 \
  else if (cfg.cw == 1) YAMOE_CASE(kWarp, 1, M)                                  \
  else if (cfg.cw == 2) YAMOE_CASE(kWarp, 2, M)                                  \
  else YAMOE_CASE(kWarp, 4, M)

template <typename scalar_t, bool FP8>
void launch_all(const torch::Tensor& x, const WSrc& wgu, const WSrc& wd, const torch::Tensor& rw,
                torch::Tensor& act, torch::Tensor& y, torch::Tensor& out, const torch::Tensor& perm, const int* blk_expert,
                const int* blk_row, const int* blk_end, int T, int H, int I, int K, int E, int max_blocks, int bm, Launch up,
                Launch dn, cudaStream_t stream) {
  const dim3 grid_up(max_blocks, (I + up.cols - 1) / up.cols);
  const size_t smem_up = (size_t)bm * H * sizeof(scalar_t);
#define YAMOE_UP(L)                                                                                          \
  gate_up_silu_kernel<scalar_t, FP8, L, CWV><<<grid_up, up.threads, smem_up, stream>>>(                           \
      x.data_ptr<scalar_t>(), wgu, act.data_ptr<scalar_t>(), perm.data_ptr<int>(),                            \
      blk_expert, blk_row, blk_end, H, I, K, E, bm, up.cols, up.mfma_min, up.act)
  YAMOE_SWITCH(up, YAMOE_UP);
#undef YAMOE_UP

  const dim3 grid_dn(max_blocks, (H + dn.cols - 1) / dn.cols);
  const size_t smem_dn = (size_t)bm * I * sizeof(scalar_t);
#define YAMOE_DN(L)                                                                                          \
  down_kernel<scalar_t, FP8, L, CWV><<<grid_dn, dn.threads, smem_dn, stream>>>(                                   \
      act.data_ptr<scalar_t>(), wd, rw.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),                          \
      perm.data_ptr<int>(), blk_expert, blk_row, blk_end, H, I, E, bm, dn.cols, dn.mfma_min)
  YAMOE_SWITCH(dn, YAMOE_DN);
#undef YAMOE_DN

  const int64_t n = (int64_t)T * H;
  topk_sum_kernel<scalar_t><<<(n + 255) / 256, 256, 0, stream>>>(y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                                                                T, K, H);
}

}  // namespace

torch::Tensor fused_experts(torch::Tensor const& hidden_states,  // [T, H]
                            torch::Tensor const& top_k_index,    // [T, K] int
                            torch::Tensor const& top_k_weights,  // [T, K]
                            torch::Tensor const& gate_up_proj,   // [E, 2I, H]
                            torch::Tensor const& down_proj,      // [E, H, I]
                            int64_t threads_up, int64_t cols_up, int64_t lanes_up, int64_t cw_up, int64_t threads_down,
                            int64_t cols_down, int64_t lanes_down, int64_t cw_down, int64_t mfma_min,
                            double tile_min_rows, double tile16_max_rows, int64_t act_kind,
                            c10::optional<torch::Tensor> const& gate_up_scale,  // FP8: [E, ceil(2I / bn), ceil(H / bk)]
                            c10::optional<torch::Tensor> const& down_scale,     // FP8: [E, ceil(H / bn), ceil(I / bk)]
                            int64_t block_n, int64_t block_k) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dim() == 2, "hidden_states must be a 2D GPU tensor");
  TORCH_CHECK(gate_up_proj.dim() == 3 && down_proj.dim() == 3, "expert weights must be 3D");
  // Weights are either the activation dtype or FP8 e4m3fn with fp32 block scales (weight-only dequant).
  const bool fp8 = gate_up_proj.scalar_type() == at::kFloat8_e4m3fn;
  TORCH_CHECK(down_proj.scalar_type() == gate_up_proj.scalar_type(), "gate_up_proj and down_proj must share a dtype");
  TORCH_CHECK(fp8 || hidden_states.scalar_type() == gate_up_proj.scalar_type(),
              "expert weights must match hidden_states dtype or be float8_e4m3fn");
  const int T = hidden_states.size(0), H = hidden_states.size(1);
  const int K = top_k_index.size(1);
  const int E = gate_up_proj.size(0), I = gate_up_proj.size(1) / 2;
  TORCH_CHECK(gate_up_proj.size(2) == H && down_proj.size(0) == E && down_proj.size(1) == H &&
                  down_proj.size(2) == I,
              "expected gate_up_proj [E, 2I, H] and down_proj [E, H, I]");
  TORCH_CHECK(E <= kMaxExperts, "at most ", kMaxExperts, " experts supported");
  TORCH_CHECK(act_kind == kActSilu || act_kind == kActGeluTanh, "act_kind must be 0 (silu) or 1 (gelu tanh)");
  TORCH_CHECK(H % 8 == 0 && I % 8 == 0, "H and I must be multiples of 8");
  // Rows per block: largest power of two <= kMaxM whose staging buffer fits the budget.
  const int64_t row_bytes = (int64_t)std::max(H, I) * hidden_states.element_size();
  int bm = kMaxM;
  while (bm > 1 && bm * row_bytes > kSmemBudget) bm /= 2;
  TORCH_CHECK(bm * row_bytes <= kSmemBudget, "hidden size too large for the shared-memory staging buffer");
  for (int64_t l : {lanes_up, lanes_down})
    TORCH_CHECK(l == 8 || l == 16 || l == 32 || l == kWarp, "lanes per column must be 8, 16, 32 or the warp size");
  TORCH_CHECK(threads_up <= kMaxThreads && threads_down <= kMaxThreads, "too many threads");
  if (fp8) {
    TORCH_CHECK(gate_up_scale.has_value() && down_scale.has_value(), "FP8 weights need gate_up_scale and down_scale");
    TORCH_CHECK(block_n > 0 && block_k > 0 && block_k % 16 == 0, "FP8 block_k must be a positive multiple of 16");
    TORCH_CHECK(H % 16 == 0 && I % 16 == 0, "FP8 weights need H and I to be multiples of 16");
    const auto& gs = *gate_up_scale;
    const auto& ds = *down_scale;
    auto cdiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    TORCH_CHECK(gs.scalar_type() == at::kFloat && ds.scalar_type() == at::kFloat, "FP8 scales must be float32");
    TORCH_CHECK(gs.dim() == 3 && gs.size(0) == E && gs.size(1) == cdiv(2 * I, block_n) && gs.size(2) == cdiv(H, block_k),
                "gate_up_scale must be [E, ceil(2I / block_n), ceil(H / block_k)]");
    TORCH_CHECK(ds.dim() == 3 && ds.size(0) == E && ds.size(1) == cdiv(H, block_n) && ds.size(2) == cdiv(I, block_k),
                "down_scale must be [E, ceil(H / block_n), ceil(I / block_k)]");
  }

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto x = hidden_states.contiguous();
  auto wgu = gate_up_proj.contiguous();
  auto wd = down_proj.contiguous();
  auto ids = top_k_index.reshape({-1}).to(torch::kInt64).contiguous();
  auto rw = top_k_weights.reshape({-1}).to(x.scalar_type()).contiguous();
  torch::Tensor gs_c, ds_c;
  if (fp8) gs_c = gate_up_scale->contiguous(), ds_c = down_scale->contiguous();
  const WSrc src_gu{wgu.data_ptr(), fp8 ? gs_c.data_ptr<float>() : nullptr, fp8 ? (int)gs_c.size(1) : 0,
                    fp8 ? (int)gs_c.size(2) : 0, (int)block_n, (int)block_k};
  const WSrc src_dn{wd.data_ptr(), fp8 ? ds_c.data_ptr<float>() : nullptr, fp8 ? (int)ds_c.size(1) : 0,
                    fp8 ? (int)ds_c.size(2) : 0, (int)block_n, (int)block_k};

  const int S = T * K;
  // Matrix-core paths need MFMA (ROCm) or mma.sync bf16 / f16 (sm80+).
#ifdef YAMOE_AMD
  const bool has_mma = true;
#else
  const bool has_mma = at::cuda::getCurrentDeviceProperties()->major >= 8;
#endif
  if (!has_mma) mfma_min = std::numeric_limits<int>::max();
  // Average rows per expert decides the path: GEMV / matrix-core decode kernel below tile_min_rows, then
  // 16-row tiles, then 64-row tiles once experts fill up, so each expert's weights are read once per tile.
  if (has_mma && (double)S >= tile_min_rows * E && H % 32 == 0 && I % 32 == 0) {
    const int tm = (double)S <= tile16_max_rows * E ? 16 : kTileM;
    const int tb = std::min(S, E) + S / tm;
    auto iopt_t = x.options().dtype(torch::kInt32);
    auto perm_t = torch::empty({S}, iopt_t);
    auto blk_t = torch::empty({3, tb}, iopt_t);
    int* be = blk_t[0].data_ptr<int>();
    int* br = blk_t[1].data_ptr<int>();
    int* bn = blk_t[2].data_ptr<int>();
    align_kernel<<<1, kMaxExperts, 0, stream>>>(ids.data_ptr<int64_t>(), S, E, tb, tm, perm_t.data_ptr<int>(),
                                                be, br, bn);
    auto act_t = torch::empty({S, I}, x.options());
    auto y_t = torch::empty({S, H}, x.options());
    auto out_t = torch::empty({T, H}, x.options());
    const int64_t n = (int64_t)T * H;
    auto run = [&](auto tag, auto fp8_tag) {
      using scalar_t = decltype(tag);
      constexpr bool FP8V = decltype(fp8_tag)::value;
      auto launch = [&](auto tm_tag, auto bk_tag) {
        constexpr int TMV = decltype(tm_tag)::value;
        constexpr int BKV = decltype(bk_tag)::value;
        constexpr int TN = tile_n<TMV>();
        const dim3 grid_up(tb, (I + TN - 1) / TN), grid_dn(tb, (H + TN - 1) / TN);
        moe_tile_kernel<scalar_t, FP8V, true, TMV, BKV><<<grid_up, kTileThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), src_gu, act_t.data_ptr<scalar_t>(), nullptr,
            perm_t.data_ptr<int>(), be, br, bn, I, H, K, E, (int)act_kind);
        moe_tile_kernel<scalar_t, FP8V, false, TMV, BKV><<<grid_dn, kTileThreads, 0, stream>>>(
            act_t.data_ptr<scalar_t>(), src_dn, y_t.data_ptr<scalar_t>(), rw.data_ptr<scalar_t>(),
            perm_t.data_ptr<int>(), be, br, bn, H, I, 1, E, (int)act_kind);
      };
      if (tm == 16 && H % 64 == 0 && I % 64 == 0) launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 64>{});
      else if (tm == 16) launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 32>{});
      else launch(std::integral_constant<int, kTileM>{}, std::integral_constant<int, 32>{});
      topk_sum_kernel<scalar_t><<<(n + 255) / 256, 256, 0, stream>>>(y_t.data_ptr<scalar_t>(),
                                                                    out_t.data_ptr<scalar_t>(), T, K, H);
    };
    using T8 = std::integral_constant<bool, true>;
    using T16 = std::integral_constant<bool, false>;
    if (x.scalar_type() == at::kBFloat16) fp8 ? run(at::BFloat16{}, T8{}) : run(at::BFloat16{}, T16{});
    else if (x.scalar_type() == at::kHalf) fp8 ? run(at::Half{}, T8{}) : run(at::Half{}, T16{});
    else TORCH_CHECK(false, "fused_experts supports bfloat16 and float16");
    return out_t;
  }
  const int max_blocks = std::min(S, E) + S / bm;
  auto iopt = x.options().dtype(torch::kInt32);
  auto perm = torch::empty({S}, iopt);
  auto blk = torch::empty({3, max_blocks}, iopt);
  int* blk_expert = blk[0].data_ptr<int>();
  int* blk_row = blk[1].data_ptr<int>();
  int* blk_end = blk[2].data_ptr<int>();

  align_kernel<<<1, kMaxExperts, 0, stream>>>(ids.data_ptr<int64_t>(), S, E, max_blocks, bm, perm.data_ptr<int>(),
                                              blk_expert, blk_row, blk_end);

  auto act = torch::empty({S, I}, x.options());
  auto y = torch::empty({S, H}, x.options());
  auto out = torch::empty({T, H}, x.options());
  const Launch up{(int)threads_up, (int)cols_up, (int)lanes_up, (int)cw_up, (int)mfma_min, (int)act_kind};
  const Launch dn{(int)threads_down, (int)cols_down, (int)lanes_down, (int)cw_down, (int)mfma_min, (int)act_kind};

#define YAMOE_ALL(ST, F8)                                                                                     \
  launch_all<ST, F8>(x, src_gu, src_dn, rw, act, y, out, perm, blk_expert, blk_row, blk_end, T, H, I, K, E,     \
                     max_blocks, bm, up, dn, stream)
  if (x.scalar_type() == at::kBFloat16) {
    if (fp8) YAMOE_ALL(at::BFloat16, true); else YAMOE_ALL(at::BFloat16, false);
  } else if (x.scalar_type() == at::kHalf) {
    if (fp8) YAMOE_ALL(at::Half, true); else YAMOE_ALL(at::Half, false);
  } else {
    TORCH_CHECK(false, "fused_experts supports bfloat16 and float16");
  }
#undef YAMOE_ALL
  return out;
}
