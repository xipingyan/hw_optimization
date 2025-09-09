#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

// Reference: https://github.com/usstq/aboutSHW/blob/main/opencl/clops/linear_f16xmx.py#L80
// Learn how to use intel XMX to accelerate gemm.
#define WG_SGS 8 // sub-group size (must be 8 for dpasw)
#define WG_N 8   // max 16
#define WG_M 4   // max 8
#define SG_M (4 * 8)
#define SG_N (2 * 8)
// all sub-groups in a work-group handles sub-matrix C of shape [BM, BN]
#define BM WG_M *SG_M // 4*(4*8)
#define BN WG_N *SG_N // 8*(2*8)
#define BK 64         // BK is limited by SLM size
#define SLM_size 65536
#define SLM_use (BM * BK + BN * BK) * 2

#define USE_DPAS 0
#define USE_DPASW 1

ulong __attribute__((overloadable)) intel_get_cycle_counter(void);
__attribute__((intel_reqd_sub_group_size(WG_SGS)))
__kernel void gemm_XMX_prepackB(__global half *Bsrc, __global half *Bdst, int N, int K)
{
    int sg_c = get_local_id(0); // # sub-group local/channel id
    int sg_n = get_local_id(1); // # sub-group id in 'N' dim
    int sg_m = get_local_id(2); // # sub-group id in 'M' dim
    int sg_id = get_sub_group_id();

    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);

    // printf("lid=[%d,%d,%d], sg_id=%d, wg_id_n_m[%d,%d]\n", sg_c, sg_n, sg_m, sg_id, wg_n, wg_m);
    // if (get_global_id(0) == 0 && get_global_id(2) == 0 && get_global_id(1) == 0) {
    //     printf("** gws=[%d,%d,%d], lws[%d,%d,%d]\n", get_global_size(0), get_global_size(1), get_global_size(2), 
    //         get_local_size(0), get_local_size(1), get_local_size(2));
    // }
    // # Bsrc/Bdst: [N, K]
    Bsrc += wg_n * BN * K;
    Bdst += wg_n * BN * K;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        int x = sg_m;
        __global half *src = Bsrc + (sg_n * 16 + sg_c) * K + k0 + x * 16;
        uint8 blk0 = *(__global uint8 *)(src);
        uint8 blk1 = *(__global uint8 *)(src + 8 * K);

        __global half *dst = Bdst + k0 * BN + (sg_n * 16 * BK) + sg_m * (16 * 16);
        intel_sub_group_block_write8((__global uint *)(dst), blk0);
        intel_sub_group_block_write8((__global uint *)(dst + 8 * 16), blk1);
    }
}

// # Register resource needs to be carefully used since C matrix is resident in accumulator register only
// #   -  avoid using array since compiler would prefer to spill them first
// #   -  carefully choose which variable lives in global scope to limit register allocation life-cycle

__attribute__((intel_reqd_sub_group_size(WG_SGS)))
__kernel void
gemm_XMX_tput(__global half *A, __global half *B, __global float *bias, __global half *C, int M, int K, int N)
{
    // # groups/local : [1, N/BN, M/BM] / [WG_SGS, WG_N, WG_M]
    // # global         [8, N/SG_N, M/SG_M]

    // # local memory for A slice & B slice
    __local half buffA[BM * BK];
    __local half buffB[BN * BK];

    // # accumulators 4 x 2 register tile (each reg is 8x8 float)
    float8 c00 = 0.0f;
    float8 c10 = 0.0f;
    float8 c20 = 0.0f;
    float8 c30 = 0.0f;

    float8 c01 = 0.0f;
    float8 c11 = 0.0f;
    float8 c21 = 0.0f;
    float8 c31 = 0.0f;

    __local half *pbuffA;
    __local half *pbuffB;
    const __local uint *pA;
    const __local uint *pB;

    __global half *A_last = A + (M - 1) * K;

    {
        int sg_c = get_local_id(0); // # sub-group local/channel id
        int sg_n = get_local_id(1); // # sub-group id in 'N' dim
        int sg_m = get_local_id(2); // # sub-group id in 'M' dim
        int sg_id = get_sub_group_id();

        int wg_n = get_group_id(1);
        int wg_m = get_group_id(2);
        // # A: [M, K]
        // # B: [N, K]
        B += wg_n * BN * K + (sg_n * 16 * BK) + sg_m * (16 * 16);

        pbuffB = buffB + (sg_n * 16 * BK) + sg_m * (16 * 16);

        int m = wg_m * BM + sg_m * 32 + sg_n * (32 / WG_N);
        if (m >= M)
            m = M - 1; // # avoid overflow
        A += m * K;
        pbuffA = buffA + (sg_m * 32 * BK) + (sg_n * (32 / WG_N)) * 16;

#if USE_DPASW
        pA = (__local uint *)(buffA + (sg_m * 32 * BK) + (sg_id & 1) * (4 * 16));
#elif USE_DPAS
        pA = (__local uint *)(buffA + (sg_m * 32 * BK));
#endif
        pB = (const __local uint *)(buffB + (sg_n * 16) * BK);
    }

    // # outer loop in 'K' dim
    for (int k0 = 0; k0 < K; k0 += BK)
    {
// # commented to test performance of [SLM + dpasw] only
#if 1
        // # load & pack A into buffA:
        {
            __global half *src = A + k0;
            uint4 r0 = intel_sub_group_block_read4((__global uint *)(src));
            if (src < A_last)
                src += K;
            uint4 r1 = intel_sub_group_block_read4((__global uint *)(src));
            if (src < A_last)
                src += K;
            uint4 r2 = intel_sub_group_block_read4((__global uint *)(src));
            if (src < A_last)
                src += K;
            uint4 r3 = intel_sub_group_block_read4((__global uint *)(src));

            uint4 a0 = (uint4)(r0.s0, r1.s0, r2.s0, r3.s0);
            uint4 a1 = (uint4)(r0.s1, r1.s1, r2.s1, r3.s1);
            uint4 a2 = (uint4)(r0.s2, r1.s2, r2.s2, r3.s2);
            uint4 a3 = (uint4)(r0.s3, r1.s3, r2.s3, r3.s3);

            intel_sub_group_block_write4((__local uint *)(pbuffA), a0);
            intel_sub_group_block_write4((__local uint *)(pbuffA + 32 * 16 * 1), a1);
            intel_sub_group_block_write4((__local uint *)(pbuffA + 32 * 16 * 2), a2);
            intel_sub_group_block_write4((__local uint *)(pbuffA + 32 * 16 * 3), a3);
        }
#endif
#if 1
        // # load sub-slice B (256*BK) into buffB : (256*BK)/(16*8) = 128 half (8 x regs)
        // #  suppose B has been turned into blocked layout : [K/BK, 256, BK]
        {
            __global half *src = B + k0 * BN;
            uint8 blk0 = intel_sub_group_block_read8((__global uint *)(src));
            uint8 blk1 = intel_sub_group_block_read8((__global uint *)(src + 8 * 16));
            intel_sub_group_block_write8((__local uint *)(pbuffB), blk0);
            intel_sub_group_block_write8((__local uint *)(pbuffB + 8 * 16), blk1);
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
#if 1
        // # offset into buff A & B for each sub-group
        // # sub-group 8-rows, 16-cols: each of size 32x16
        // # both are blocked layout:
        // #        each SG in a row share 32xBK buffA slice
        // #        each SG in a col share 16xBK buffB slice

        // # unroll would cause register exhaast & spill
        __attribute__((opencl_unroll_hint(1))) for (int k1 = 0; k1 < BK / 16; k1++)
        {
            // # limit scopes of temp regs to reuse-registers
#if USE_DPASW
            int4 a0 = as_int4(intel_sub_group_block_read4(pA + 0 * 8 * 8));
            int4 a1 = as_int4(intel_sub_group_block_read4(pA + 1 * 8 * 8));
            int4 a2 = as_int4(intel_sub_group_block_read4(pA + 2 * 8 * 8));
            int4 a3 = as_int4(intel_sub_group_block_read4(pA + 3 * 8 * 8));

            // # scatter load is no good here, SLM will be in best tput using blocked sub-group read
            int8 b0 = as_int8(intel_sub_group_block_read8(pB + 0 * 8 * 8));
            int8 b1 = as_int8(intel_sub_group_block_read8(pB + 1 * 8 * 8));

            c00 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c00);
            c10 = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b0, c10);
            c20 = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b0, c20);
            c30 = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b0, c30);

            c01 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b1, c01);
            c11 = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b1, c11);
            c21 = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b1, c21);
            c31 = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b1, c31);
#elif USE_DPAS
            int8 a0 = as_int8(intel_sub_group_block_read8(pA + 0 * 8 * 8));
            int8 a1 = as_int8(intel_sub_group_block_read8(pA + 1 * 8 * 8));
            int8 a2 = as_int8(intel_sub_group_block_read8(pA + 2 * 8 * 8));
            int8 a3 = as_int8(intel_sub_group_block_read8(pA + 3 * 8 * 8));

            // # scatter load is no good here, SLM will be in best tput using blocked sub-group read
            int8 b0 = as_int8(intel_sub_group_block_read8(pB + 0 * 8 * 8));
            int8 b1 = as_int8(intel_sub_group_block_read8(pB + 1 * 8 * 8));

            c00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c00);
            c10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, c10);
            c20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, c20);
            c30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, c30);

            c01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, c01);
            c11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, c11);
            c21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, c21);
            c31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, c31);
#endif
            pA += 4 * 8 * 8;
            pB += 2 * 8 * 8;
        }
        // # rewind pA & pB after loop to save registers
        pA -= BK / 16 * 4 * 8 * 8;
        pB -= BK / 16 * 2 * 8 * 8;
#endif
        // # need to sync again at the end, to avoid faster threads
        // # fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // # store sub-C
    {
        int sg_c = get_local_id(0); // # sub-group local/channel id
        int sg_n = get_local_id(1); // # sub-group id in 'N' dim
        int sg_m = get_local_id(2); // # sub-group id in 'M' dim
        int sg_id = get_sub_group_id();

        int n = get_group_id(1) * BN + sg_n * 16;
        int m = get_group_id(2) * BM + sg_m * 32;
        int channel = get_sub_group_local_id();

        if (bias)
        {
            float8 bias0 = convert_float8((half8)(bias[n + channel]));
            float8 bias1 = convert_float8((half8)(bias[n + 8 + channel]));
            c00 += bias0;
            c01 += bias1;
            c10 += bias0;
            c11 += bias1;
            c20 += bias0;
            c21 += bias1;
            c30 += bias0;
            c31 += bias1;
        }

        // # transpose with the help of SLM : using sub-group/scatter load/store
        // # store to SLM in compact format
        // # scatter reads & scatter write to change strides from 8(compat) to N(strided)
        __local half *scratch = buffA + get_sub_group_id() * 8 * 8 * 2;
        __local half *src = (__local half *)(scratch + channel * 8);
#if COMBINE_GATE_UP == 1
        N = N / 2;
        n = n / 2;
#endif

#if COMBINE_GATE_UP == 1
#define STORE_C_REG(creg0, creg1)                                                             \
    {                                                                                         \
        if (m >= M)                                                                           \
            m = M - 1; /*# avoid overflow*/                                                   \
        intel_sub_group_block_write_us8(scratch, as_ushort8(convert_half8(creg0)));           \
        intel_sub_group_block_write_us8(scratch + 8 * 8, as_ushort8(convert_half8(creg1)));   \
        half8 v00 = *(__local half8 *)(src);                                                  \
        half8 v01 = *(__local half8 *)(src + 8 * 8);                                          \
        __global half *dst = (__global half *)(C + m * N + n);                                \
        v00 = v00 / ((half8)(1.0f) + native_exp(-v00)) * v01; /* silu(gate_proj) * up_proj */ \
        *(__global half8 *)dst = v00;                                                         \
    }
#else
#define STORE_C_REG(creg0, creg1)                                                           \
    {                                                                                       \
        if (m >= M)                                                                         \
            m = M - 1; /*# avoid overflow*/                                                 \
        intel_sub_group_block_write_us8(scratch, as_ushort8(convert_half8(creg0)));         \
        intel_sub_group_block_write_us8(scratch + 8 * 8, as_ushort8(convert_half8(creg1))); \
        half8 v00 = *(__local half8 *)(src);                                                \
        half8 v01 = *(__local half8 *)(src + 8 * 8);                                        \
        __global half *dst = (__global half *)(C + m * N + n);                              \
        *(__global half8 *)dst = v00;                                                       \
        *(__global half8 *)(dst + 8) = v01;                                                 \
    }
#endif
        m += channel;
        STORE_C_REG(c00, c01);
        m += 8;
        STORE_C_REG(c10, c11);
        m += 8;
        STORE_C_REG(c20, c21);
        m += 8;
        STORE_C_REG(c30, c31);
    }
}