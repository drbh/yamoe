// csrc/fused_experts.cu
//
// MoE experts forward for the transformers experts layout, written once for CUDA and ROCm (hipified by
// kernel-builder). Weights are K-contiguous:
//   gate_up_proj [E, 2I, H]   rows [0, I) gate, [I, 2I) up
//   down_proj    [E, H, I]
//
//   out[t] = sum_k w[t, k] * down[e](silu(gate[e] x[t]) * up[e] x[t]),   e = top_k_index[t, k]
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
//     staged in shared memory; on ROCm, blocks with >= mfma_min rows use a 16x16 MFMA tile instead.
//   * prefill (ROCm): grouped GEMM over 16-row (BK 64) or 64-row (BK 32) blocks per expert, operands
//     staged in LDS with register prefetch, MFMA 16x16x16 bf16 / f16.
// On CUDA only the decode path is built; yamoe.experts dispatches larger inputs to grouped_mm.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>
#include <type_traits>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#define YAMOE_AMD 1
constexpr int kWarp = 64;
#define SHFL_XOR(v, o) __shfl_xor((v), (o))
#else
constexpr int kWarp = 32;
#define SHFL_XOR(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
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

#ifdef YAMOE_AMD
// ---------------------------------------------------------------------------------------------
// Matrix-core (MFMA) tile for blocks with several token rows. One wavefront computes a 16-column
// tile of out[m, n] = sum_k X[m, k] * W[n, k] with mfma_f32_16x16x16 (bf16 / f16).
//
// Operand layout for 16x16x16 (lane l, r = l % 16, g = l / 16):
//   A[r][4g + j], B[4g + j][r] (j < 4), D[4g + j][r]
// Each lane loads 16 contiguous bytes per operand and feeds them to two MFMAs, so the logical k of
// the instruction maps to physical k0 + 8g + j (first MFMA) and k0 + 8g + 4 + j (second). A and B use
// the same mapping, so the reduction is exact. Rows r >= m feed zeros.
// ---------------------------------------------------------------------------------------------
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

template <typename scalar_t>
__device__ __forceinline__ yamoe_f32x4 mfma32(uint4 a, uint4 b, yamoe_f32x4 c) {
  c = mfma16<scalar_t>(make_uint2(a.x, a.y), make_uint2(b.x, b.y), c);
  return mfma16<scalar_t>(make_uint2(a.z, a.w), make_uint2(b.z, b.w), c);
}

__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }

// gate/up + silu over a 16-column tile per wavefront. Requires H % 64 == 0.
template <typename scalar_t>
__device__ __forceinline__ void gate_up_mfma(const scalar_t* xs, const scalar_t* __restrict__ wg_e,
                                             scalar_t* __restrict__ act, const int* pairs, int m, int H,
                                             int I, int c0, int c1) {
  const int wave = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, nwaves = blockDim.x / kWarp;
  const int r = lane % 16, g = lane / 16;
  const bool arow = r < m;
  const scalar_t* xr = xs + r * H + 8 * g;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  for (int cb = c0 + wave * 16; cb < c1; cb += nwaves * 16) {
    const scalar_t* wg = wg_e + (int64_t)min(cb + r, c1 - 1) * H + 8 * g;
    const scalar_t* wu = wg + (int64_t)I * H;
    yamoe_f32x4 ag = {0.f, 0.f, 0.f, 0.f}, au = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < H; k += 64) {
      const uint4 g0 = ld16(wg + k), g1 = ld16(wg + k + 32);
      const uint4 u0 = ld16(wu + k), u1 = ld16(wu + k + 32);
      const uint4 x0 = arow ? ld16(xr + k) : zero, x1 = arow ? ld16(xr + k + 32) : zero;
      ag = mfma32<scalar_t>(x0, g0, ag);
      au = mfma32<scalar_t>(x0, u0, au);
      ag = mfma32<scalar_t>(x1, g1, ag);
      au = mfma32<scalar_t>(x1, u1, au);
    }
    const int c = cb + r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = 4 * g + j;
      if (row < m && c < c1) {
        const float gv = ag[j], uv = au[j];
        act[(int64_t)pairs[row] * I + c] = static_cast<scalar_t>(gv / (1.f + __expf(-gv)) * uv);
      }
    }
  }
}

// down projection * routing weight over a 16-column tile per wavefront. Requires I % 64 == 0.
template <typename scalar_t>
__device__ __forceinline__ void down_mfma(const scalar_t* as, const scalar_t* __restrict__ wd_e,
                                          const scalar_t* __restrict__ rw, scalar_t* __restrict__ y,
                                          const int* pairs, int m, int H, int I, int c0, int c1) {
  const int wave = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, nwaves = blockDim.x / kWarp;
  const int r = lane % 16, g = lane / 16;
  const bool arow = r < m;
  const scalar_t* ar = as + r * I + 8 * g;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  for (int cb = c0 + wave * 16; cb < c1; cb += nwaves * 16) {
    const scalar_t* wd = wd_e + (int64_t)min(cb + r, c1 - 1) * I + 8 * g;
    yamoe_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < I; k += 64) {
      const uint4 w0 = ld16(wd + k), w1 = ld16(wd + k + 32);
      const uint4 a0 = arow ? ld16(ar + k) : zero, a1 = arow ? ld16(ar + k + 32) : zero;
      acc = mfma32<scalar_t>(a0, w0, acc);
      acc = mfma32<scalar_t>(a1, w1, acc);
    }
    const int c = cb + r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = 4 * g + j;
      if (row < m && c < c1) {
        const int p = pairs[row];
        y[(int64_t)p * H + c] = static_cast<scalar_t>(acc[j] * static_cast<float>(rw[p]));
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Prefill tiles: grouped GEMM over kTileM-row blocks of one expert x kTileN output columns.
// 4 wavefronts in a 2x2 grid, each owning a 32x32 sub-tile (2x2 MFMA 16x16 tiles). Operands are
// loaded straight from global memory in the same permuted 16-byte layout as the decode tile.
// ---------------------------------------------------------------------------------------------
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileThreads = 256;

// One kernel for both projections:
//   GATED  (gate/up): A row = x[pair / topk], W rows c (gate) and N + c (up), out = act[pair, c] = silu(g) * u
//   !GATED (down):    A row = act[pair],      W row c,                      out = y[pair, c]   = acc * rw[pair]
// Per 32-wide k-step the CTA stages a 64-row A tile and 64-row W tile(s) in LDS (one 16-byte load per thread
// per tile) and prefetches the next step into registers while the matrix cores work.

// TM: rows per block (64 or 16). Waves are arranged (TM / WM) x (kTileN / WN) = 4 with WM = min(TM, 32).
// BK: k-step (32 or 64). Each thread loads BK / 32 16-byte chunks per tile per step.
template <typename scalar_t, bool GATED, int TM, int BK>
__global__ void __launch_bounds__(kTileThreads) moe_tile_kernel(
    const scalar_t* __restrict__ a_src, const scalar_t* __restrict__ w, scalar_t* __restrict__ out,
    const scalar_t* __restrict__ rw, const int* __restrict__ perm, const int* __restrict__ blk_expert,
    const int* __restrict__ blk_row, const int* __restrict__ blk_end, int N, int Kd, int adiv, int E) {
  constexpr int kLds = BK + 8;  // padded LDS row (elements), keeps 16-byte alignment
  constexpr int NC = BK / 32;    // 16-byte chunks per thread per tile row
  constexpr int WM = TM < 32 ? TM : 32;
  constexpr int WN = kTileN * TM / (WM * 4);  // (TM / WM) row waves x (kTileN / WN) col waves = 4
  constexpr int FI = WM / 16, FJ = WN / 16;
  static_assert((TM / WM) * (kTileN / WN) == kTileThreads / 64, "wave grid must use all waves");
  __shared__ __align__(16) scalar_t As[TM * kLds];
  __shared__ __align__(16) scalar_t Bg[kTileN * kLds];
  __shared__ __align__(16) scalar_t Bu[GATED ? kTileN * kLds : 8];
  __shared__ int pairs[TM];
  const int b = blockIdx.x;
  const int e = blk_expert[b];
  if (e >= E) return;
  const int row0 = blk_row[b];
  const int m = min(blk_end[b] - row0, TM);
  const int tid = threadIdx.x;
  if (tid < TM) pairs[tid] = tid < m ? perm[row0 + tid] : 0;
  __syncthreads();

  // loader: thread -> (row, 8-element chunk) of each 64 x 32 tile
  const int lr = tid >> 2, lc = (tid & 3) * 8;
  const int n_base = blockIdx.y * kTileN;
  const bool a_ld_thread = lr < TM;  // only the first TM * 4 threads load A
  const bool a_ok = a_ld_thread && lr < m;
  const scalar_t* a_ld = a_src + (int64_t)(pairs[a_ok ? lr : 0] / adiv) * Kd + lc;
  const scalar_t* w_e = w + (int64_t)e * (GATED ? 2 : 1) * N * Kd;
  const scalar_t* g_ld = w_e + (int64_t)min(n_base + lr, N - 1) * Kd + lc;
  const scalar_t* u_ld = g_ld + (int64_t)N * Kd;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  uint4 ra[NC], rg[NC], ru[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    ra[c] = a_ok ? ld16(a_ld + 32 * c) : zero;
    rg[c] = ld16(g_ld + 32 * c);
    ru[c] = GATED ? ld16(u_ld + 32 * c) : zero;
  }

  const int wave = tid / kWarp, lane = tid % kWarp;
  const int r = lane % 16, g = lane / 16;
  constexpr int WAVES_N = kTileN / WN;
  const int wr = (wave / WAVES_N) * WM, wc = (wave % WAVES_N) * WN;
  yamoe_f32x4 ag[FI][FJ], au[FI][FJ];
#pragma unroll
  for (int i = 0; i < FI; ++i)
#pragma unroll
    for (int j = 0; j < FJ; ++j) ag[i][j] = au[i][j] = yamoe_f32x4{0.f, 0.f, 0.f, 0.f};

  for (int k = 0; k < Kd; k += BK) {
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      if (a_ld_thread) *reinterpret_cast<uint4*>(&As[lr * kLds + lc + 32 * c]) = ra[c];
      *reinterpret_cast<uint4*>(&Bg[lr * kLds + lc + 32 * c]) = rg[c];
      if (GATED) *reinterpret_cast<uint4*>(&Bu[lr * kLds + lc + 32 * c]) = ru[c];
    }
    __syncthreads();
    if (k + BK < Kd) {
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        ra[c] = a_ok ? ld16(a_ld + k + BK + 32 * c) : zero;
        rg[c] = ld16(g_ld + k + BK + 32 * c);
        if (GATED) ru[c] = ld16(u_ld + k + BK + 32 * c);
      }
    }
#pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      uint4 fa[FI], fg[FJ], fu[FJ];
#pragma unroll
      for (int i = 0; i < FI; ++i) fa[i] = *reinterpret_cast<const uint4*>(&As[(wr + i * 16 + r) * kLds + kk + 8 * g]);
#pragma unroll
      for (int j = 0; j < FJ; ++j) {
        fg[j] = *reinterpret_cast<const uint4*>(&Bg[(wc + j * 16 + r) * kLds + kk + 8 * g]);
        if (GATED) fu[j] = *reinterpret_cast<const uint4*>(&Bu[(wc + j * 16 + r) * kLds + kk + 8 * g]);
      }
#pragma unroll
      for (int i = 0; i < FI; ++i)
#pragma unroll
        for (int j = 0; j < FJ; ++j) {
          ag[i][j] = mfma32<scalar_t>(fa[i], fg[j], ag[i][j]);
          if (GATED) au[i][j] = mfma32<scalar_t>(fa[i], fu[j], au[i][j]);
        }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < FI; ++i)
#pragma unroll
    for (int j = 0; j < FJ; ++j) {
      const int c = n_base + wc + j * 16 + r;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int row = wr + i * 16 + 4 * g + q;
        if (row < m && c < N) {
          const int p = pairs[row];
          float v;
          if (GATED) {
            const float gv = ag[i][j][q], uv = au[i][j][q];
            v = gv / (1.f + __expf(-gv)) * uv;
          } else {
            v = ag[i][j][q] * static_cast<float>(rw[p]);
          }
          out[(int64_t)p * N + c] = static_cast<scalar_t>(v);
        }
      }
    }
}
#endif  // YAMOE_AMD

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
  __shared__ int scan_rows[kMaxExperts];
  __shared__ int scan_blks[kMaxExperts];
  const int tid = threadIdx.x;

  cnt[tid] = 0;
  cursor[tid] = 0;
  for (int b = tid; b < max_blocks; b += blockDim.x) blk_expert[b] = E;
  __syncthreads();
  for (int p = tid; p < S; p += blockDim.x) atomicAdd(&cnt[(int)ids[p]], 1);
  __syncthreads();

  // inclusive Hillis-Steele scan of rows and blocks per expert
  const int c = tid < E ? cnt[tid] : 0;
  const int nb = (c + bm - 1) / bm;
  scan_rows[tid] = c;
  scan_blks[tid] = nb;
  __syncthreads();
  for (int o = 1; o < blockDim.x; o <<= 1) {
    const int r = tid >= o ? scan_rows[tid - o] : 0;
    const int b = tid >= o ? scan_blks[tid - o] : 0;
    __syncthreads();
    scan_rows[tid] += r;
    scan_blks[tid] += b;
    __syncthreads();
  }
  const int start = scan_rows[tid] - c;   // exclusive
  const int bstart = scan_blks[tid] - nb;

  for (int p = tid; p < S; p += blockDim.x) {
    const int e = (int)ids[p];
    perm[scan_rows[e] - cnt[e] + atomicAdd(&cursor[e], 1)] = p;
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
template <typename scalar_t, int M, int LANES, int COLS>
__device__ __forceinline__ void gate_up_cols(const scalar_t* xs, const scalar_t* __restrict__ wg_e,
                                             scalar_t* __restrict__ act, const int* pairs, int m,
                                             int H, int I, int c0, int c1) {
  // Each group of LANES lanes computes COLS adjacent output columns at once, so every activation
  // vector read from shared memory is reused COLS times (matters when M > 1).
  const int group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
  const int ngroups = blockDim.x / LANES;
  const int vecs = H / 8;
  for (int cb = c0 + group * COLS; cb < c1; cb += ngroups * COLS) {
    const scalar_t* wg[COLS];
    const scalar_t* wu[COLS];
#pragma unroll
    for (int j = 0; j < COLS; ++j) {
      const int c = min(cb + j, c1 - 1);
      wg[j] = wg_e + (int64_t)c * H;
      wu[j] = wg[j] + (int64_t)I * H;
    }
    float ag[COLS][M], au[COLS][M];
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) ag[j][r] = 0.f, au[j][r] = 0.f;
    for (int v = lane; v < vecs; v += LANES) {
      float g8[COLS][8], u8[COLS][8];
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
        load8(wg[j] + v * 8, g8[j]);
        load8(wu[j] + v * 8, u8[j]);
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
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const float g = group_sum<LANES>(ag[j][r]);
        const float u = group_sum<LANES>(au[j][r]);
        if (lane == 0 && r < m && cb + j < c1)
          act[(int64_t)pairs[r] * I + cb + j] = static_cast<scalar_t>(g / (1.f + __expf(-g)) * u);
      }
  }
}

template <typename scalar_t, int LANES, int CW>
__global__ void __launch_bounds__(kMaxThreads) gate_up_silu_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ w, scalar_t* __restrict__ act,
    const int* __restrict__ perm, const int* __restrict__ blk_expert, const int* __restrict__ blk_row,
    const int* __restrict__ blk_end, int H, int I, int topk, int E, int bm, int cols, int mfma_min) {
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
  const scalar_t* wg_e = w + (int64_t)e * 2 * I * H;
#ifdef YAMOE_AMD
  if (m >= mfma_min && H % 64 == 0) {
    gate_up_mfma<scalar_t>(xs, wg_e, act, pairs, m, H, I, c0, c1);
    return;
  }
#endif
  if (m == 1)      gate_up_cols<scalar_t, 1, LANES, 1>(xs, wg_e, act, pairs, m, H, I, c0, c1);
  else if (m == 2) gate_up_cols<scalar_t, 2, LANES, CW>(xs, wg_e, act, pairs, m, H, I, c0, c1);
  else if (m <= 4) gate_up_cols<scalar_t, 4, LANES, CW>(xs, wg_e, act, pairs, m, H, I, c0, c1);
  else             gate_up_cols<scalar_t, kMaxM, LANES, CW>(xs, wg_e, act, pairs, m, H, I, c0, c1);
}

// ---------------------------------------------------------------------------------------------
// 3. down projection * routing weight. grid (max_blocks, ceil(H / cols)).
// ---------------------------------------------------------------------------------------------
template <typename scalar_t, int M, int LANES, int COLS>
__device__ __forceinline__ void down_cols(const scalar_t* as, const scalar_t* __restrict__ wd_e,
                                          const scalar_t* __restrict__ rw, scalar_t* __restrict__ y,
                                          const int* pairs, int m, int H, int I, int c0, int c1) {
  const int group = threadIdx.x / LANES, lane = threadIdx.x % LANES;
  const int ngroups = blockDim.x / LANES;
  const int vecs = I / 8;
  for (int cb = c0 + group * COLS; cb < c1; cb += ngroups * COLS) {
    const scalar_t* wd[COLS];
#pragma unroll
    for (int j = 0; j < COLS; ++j) wd[j] = wd_e + (int64_t)min(cb + j, c1 - 1) * I;
    float acc[COLS][M];
#pragma unroll
    for (int j = 0; j < COLS; ++j)
#pragma unroll
      for (int r = 0; r < M; ++r) acc[j][r] = 0.f;
    for (int v = lane; v < vecs; v += LANES) {
      float w8[COLS][8];
#pragma unroll
      for (int j = 0; j < COLS; ++j) load8(wd[j] + v * 8, w8[j]);
#pragma unroll
      for (int r = 0; r < M; ++r) {
        float a8[8];
        load8(as + r * I + v * 8, a8);
#pragma unroll
        for (int j = 0; j < COLS; ++j) acc[j][r] += dot8(w8[j], a8);
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

template <typename scalar_t, int LANES, int CW>
__global__ void __launch_bounds__(kMaxThreads) down_kernel(
    const scalar_t* __restrict__ act, const scalar_t* __restrict__ w, const scalar_t* __restrict__ rw,
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
  const scalar_t* wd_e = w + (int64_t)e * H * I;
#ifdef YAMOE_AMD
  if (m >= mfma_min && I % 64 == 0) {
    down_mfma<scalar_t>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
    return;
  }
#endif
  if (m == 1)      down_cols<scalar_t, 1, LANES, 1>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
  else if (m == 2) down_cols<scalar_t, 2, LANES, CW>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
  else if (m <= 4) down_cols<scalar_t, 4, LANES, CW>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
  else             down_cols<scalar_t, kMaxM, LANES, CW>(as, wd_e, rw, y, pairs, m, H, I, c0, c1);
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
  int threads, cols, lanes, cw, mfma_min;
};

// Dispatch runtime (lanes, cw) onto template instances.
#define YAMOE_CASE(L, C, M) { constexpr int CWV = C; M(L); }
#define YAMOE_SWITCH(cfg, M)                                                     \
  if (cfg.lanes == 16 && cfg.cw == 1) YAMOE_CASE(16, 1, M)                       \
  else if (cfg.lanes == 16 && cfg.cw == 2) YAMOE_CASE(16, 2, M)                  \
  else if (cfg.lanes == 16) YAMOE_CASE(16, 4, M)                                 \
  else if (cfg.lanes == 32 && cfg.cw == 1) YAMOE_CASE(32, 1, M)                  \
  else if (cfg.lanes == 32 && cfg.cw == 2) YAMOE_CASE(32, 2, M)                  \
  else if (cfg.lanes == 32) YAMOE_CASE(32, 4, M)                                 \
  else if (cfg.cw == 1) YAMOE_CASE(kWarp, 1, M)                                  \
  else if (cfg.cw == 2) YAMOE_CASE(kWarp, 2, M)                                  \
  else YAMOE_CASE(kWarp, 4, M)

template <typename scalar_t>
void launch_all(const torch::Tensor& x, const torch::Tensor& wgu, const torch::Tensor& wd, const torch::Tensor& rw,
                torch::Tensor& act, torch::Tensor& y, torch::Tensor& out, const torch::Tensor& perm, const int* blk_expert,
                const int* blk_row, const int* blk_end, int T, int H, int I, int K, int E, int max_blocks, int bm, Launch up,
                Launch dn, cudaStream_t stream) {
  const dim3 grid_up(max_blocks, (I + up.cols - 1) / up.cols);
  const size_t smem_up = (size_t)bm * H * sizeof(scalar_t);
#define YAMOE_UP(L)                                                                                          \
  gate_up_silu_kernel<scalar_t, L, CWV><<<grid_up, up.threads, smem_up, stream>>>(                                \
      x.data_ptr<scalar_t>(), wgu.data_ptr<scalar_t>(), act.data_ptr<scalar_t>(), perm.data_ptr<int>(),      \
      blk_expert, blk_row, blk_end, H, I, K, E, bm, up.cols, up.mfma_min)
  YAMOE_SWITCH(up, YAMOE_UP);
#undef YAMOE_UP

  const dim3 grid_dn(max_blocks, (H + dn.cols - 1) / dn.cols);
  const size_t smem_dn = (size_t)bm * I * sizeof(scalar_t);
#define YAMOE_DN(L)                                                                                          \
  down_kernel<scalar_t, L, CWV><<<grid_dn, dn.threads, smem_dn, stream>>>(                                        \
      act.data_ptr<scalar_t>(), wd.data_ptr<scalar_t>(), rw.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),    \
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
                            int64_t tile_min_rows, int64_t tile16_max_rows) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dim() == 2, "hidden_states must be a 2D GPU tensor");
  TORCH_CHECK(gate_up_proj.dim() == 3 && down_proj.dim() == 3, "expert weights must be 3D");
  TORCH_CHECK(hidden_states.scalar_type() == gate_up_proj.scalar_type() &&
                  hidden_states.scalar_type() == down_proj.scalar_type(),
              "hidden_states and expert weights must share a dtype");
  const int T = hidden_states.size(0), H = hidden_states.size(1);
  const int K = top_k_index.size(1);
  const int E = gate_up_proj.size(0), I = gate_up_proj.size(1) / 2;
  TORCH_CHECK(gate_up_proj.size(2) == H && down_proj.size(0) == E && down_proj.size(1) == H &&
                  down_proj.size(2) == I,
              "expected gate_up_proj [E, 2I, H] and down_proj [E, H, I]");
  TORCH_CHECK(E <= kMaxExperts, "at most ", kMaxExperts, " experts supported");
  TORCH_CHECK(H % 8 == 0 && I % 8 == 0, "H and I must be multiples of 8");
  // Rows per block: largest power of two <= kMaxM whose staging buffer fits the budget.
  const int64_t row_bytes = (int64_t)std::max(H, I) * hidden_states.element_size();
  int bm = kMaxM;
  while (bm > 1 && bm * row_bytes > kSmemBudget) bm /= 2;
  TORCH_CHECK(bm * row_bytes <= kSmemBudget, "hidden size too large for the shared-memory staging buffer");
  for (int64_t l : {lanes_up, lanes_down})
    TORCH_CHECK(l == 16 || l == 32 || l == kWarp, "lanes per column must be 16, 32 or the warp size");
  TORCH_CHECK(threads_up <= kMaxThreads && threads_down <= kMaxThreads, "too many threads");

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto x = hidden_states.contiguous();
  auto wgu = gate_up_proj.contiguous();
  auto wd = down_proj.contiguous();
  auto ids = top_k_index.reshape({-1}).to(torch::kInt64).contiguous();
  auto rw = top_k_weights.reshape({-1}).to(x.scalar_type()).contiguous();

  const int S = T * K;
#ifdef YAMOE_AMD
  // Prefill: kTileM-row grouped-GEMM tiles so each expert's weights are read once per 64 rows.
  // Average rows per expert decides the path: GEMV/MFMA decode kernel below tile_min_rows, then 16-row tiles,
  // then 64-row tiles once experts fill up.
  if ((int64_t)S >= tile_min_rows * (int64_t)E && H % 32 == 0 && I % 32 == 0) {
    const int tm = (int64_t)S <= tile16_max_rows * (int64_t)E ? 16 : kTileM;
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
    const dim3 grid_up(tb, (I + kTileN - 1) / kTileN), grid_dn(tb, (H + kTileN - 1) / kTileN);
    const int64_t n = (int64_t)T * H;
    auto run = [&](auto tag) {
      using scalar_t = decltype(tag);
      auto launch = [&](auto tm_tag, auto bk_tag) {
        constexpr int TMV = decltype(tm_tag)::value;
        constexpr int BKV = decltype(bk_tag)::value;
        moe_tile_kernel<scalar_t, true, TMV, BKV><<<grid_up, kTileThreads, 0, stream>>>(
            x.data_ptr<scalar_t>(), wgu.data_ptr<scalar_t>(), act_t.data_ptr<scalar_t>(), nullptr,
            perm_t.data_ptr<int>(), be, br, bn, I, H, K, E);
        moe_tile_kernel<scalar_t, false, TMV, BKV><<<grid_dn, kTileThreads, 0, stream>>>(
            act_t.data_ptr<scalar_t>(), wd.data_ptr<scalar_t>(), y_t.data_ptr<scalar_t>(), rw.data_ptr<scalar_t>(),
            perm_t.data_ptr<int>(), be, br, bn, H, I, 1, E);
      };
      if (tm == 16 && H % 64 == 0 && I % 64 == 0) launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 64>{});
      else if (tm == 16) launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 32>{});
      else launch(std::integral_constant<int, kTileM>{}, std::integral_constant<int, 32>{});
      topk_sum_kernel<scalar_t><<<(n + 255) / 256, 256, 0, stream>>>(y_t.data_ptr<scalar_t>(),
                                                                    out_t.data_ptr<scalar_t>(), T, K, H);
    };
    if (x.scalar_type() == at::kBFloat16) run(at::BFloat16{});
    else if (x.scalar_type() == at::kHalf) run(at::Half{});
    else TORCH_CHECK(false, "fused_experts supports bfloat16 and float16");
    return out_t;
  }
#endif
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
  const Launch up{(int)threads_up, (int)cols_up, (int)lanes_up, (int)cw_up, (int)mfma_min};
  const Launch dn{(int)threads_down, (int)cols_down, (int)lanes_down, (int)cw_down, (int)mfma_min};

  if (x.scalar_type() == at::kBFloat16) {
    launch_all<at::BFloat16>(x, wgu, wd, rw, act, y, out, perm, blk_expert, blk_row, blk_end, T, H, I, K, E,
                             max_blocks, bm, up, dn, stream);
  } else if (x.scalar_type() == at::kHalf) {
    launch_all<at::Half>(x, wgu, wd, rw, act, y, out, perm, blk_expert, blk_row, blk_end, T, H, I, K, E,
                         max_blocks, bm, up, dn, stream);
  } else {
    TORCH_CHECK(false, "fused_experts supports bfloat16 and float16");
  }
  return out;
}
