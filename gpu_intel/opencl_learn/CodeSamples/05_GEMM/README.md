# 03_DPP_algo

Try to write dpp algorithm based on OpenCL kernel.

# How to run

    cd hw_optimization/intel_gpu/opencl_learn/CodeSamples/
    mkdir build && cd build
    cmake ..
    make -j20
    ./05_GEMM/05_GEMM

# Compare with oneDNN.

```
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:6:f16:128x1 attr-zero-points:wei:6:u8:128x1,,8x1x3584:1x3584x18944,0.217041
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:6:f16:128x1 attr-zero-points:wei:6:u8:128x1 attr-post-ops:eltwise_swish:1+binary_mul:f16:5:abc,,4x1x3584:1x3584x18944,0.278076
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:6:f16:128x1 attr-zero-points:wei:6:u8:128x1 attr-post-ops:binary_add:f16:5:abc,,4x1x18944:1x18944x3584,0.253906
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 bia:f16::blocked:abc::f0_mask4 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:6:f16:128x1 attr-zero-points:wei:6:u8:128x1,,8x1x3584:1x3584x4608,0.143066
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:6:f16:128x1 attr-zero-points:wei:6:u8:128x1 attr-post-ops:binary_add:f16:5:abc,,4x1x3584:1x3584x3584,0.139893
onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,src:f16::blocked:abc::f0 wei:u4::blocked:cab::f0 dst:f16::blocked:abc::f0,attr-scratchpad:user attr-fpmath:f16:true attr-scales:wei:6:f16:128x1 attr-zero-points:wei:6:u8:128x1,,8x1x3584:1x3584x18944,0.219971
```

# Test results

GPU: B580       <br>
EU number: 160  <br>
Max group size: 1024    <br>
Preferred group size: 16       // lws：最好设置为16的倍数   <br>

M = 3, K = 3584, N = 3584   <br>

gws=[M,N,1]
lws=[1,1,1]

<br>

|   gws   |   lws    | Kernel entry                         | Time ms  | A770 Time ms  |
| --------| -------- |:------------------------------------ | :------  | :------------ |
| [M,N,1] | [1,1,1]  | gemm_ref                             | 10.3742  | 5.4987        |
| [M,N,1] | [1,1,1]  | gemm_ref_half                        | 7.3818   | 5.6788        |
| [M,N,1] | [1,1,1]  | gemm_ref_half_weight_trans           | 3.1409   | 1.2225        |
| [M,N,1] | [1,1,1]  | gemm_half4_weight_trans              | 1.9201   | 0.6306        |
| [M,N,1] | [1,32,1] | gemm_half4_weight_trans              | 0.8651   | 0.294         |

``Test command:``

```
ENABLE_HALF=0 ENABLE_WEIGHT_TRANS=0 ENABLE_HALF4=0 ./05_GEMM/05_GEMM|
```