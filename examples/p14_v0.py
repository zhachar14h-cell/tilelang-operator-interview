# P14: GEMM
# Version: v0 · 64×64×32 分块与两阶段流水
# Syntax checked only; GPU compilation and execution not verified.
import torch
import tilelang
import tilelang.language as T


def make_gemm(M, N, K):
    # FP16 输入、FP32 累加和输出；不是纯 FP32 SGEMM。
    BM, BN, BK = 64, 64, 32
    assert M > 0 and N > 0 and K > 0
    assert M % BM == 0 and N % BN == 0 and K % BK == 0

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(N // BN, M // BM, threads=128) as (bx, by):
            a_s = T.alloc_shared((BM, BK), "float16")
            b_s = T.alloc_shared((BK, BN), "float16")
            acc = T.alloc_fragment((BM, BN), "float32")

            # 整个 K 方向共同累加到同一块 FP32 寄存器。
            T.clear(acc)
            for ko in T.Pipelined(K // BK, num_stages=2):
                T.copy(A[by * BM, ko * BK], a_s)
                T.copy(B[ko * BK, bx * BN], b_s)
                # Tensor Core MMA 的具体选择由目标 GPU 和编译器决定。
                T.gemm(a_s, b_s, acc)
            T.copy(acc, C[by * BM, bx * BN])

    return tilelang.compile(main, out_idx=None, target="cuda")


def main():
    torch.manual_seed(0)
    # reference 不通过 TF32 降低乘法精度。
    torch.backends.cuda.matmul.allow_tf32 = False
    for M, N, K in [(64, 64, 32), (256, 256, 128)]:
        a = (torch.randn(M, K, device="cuda") * 0.2).half()
        b = (torch.randn(K, N, device="cuda") * 0.2).half()
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        kernel = make_gemm(M, N, K)
        kernel(a, b, c)
        # 对相同的、已经量化为 FP16 的输入计算 reference。
        ref = a.float() @ b.float()
        torch.testing.assert_close(c, ref, rtol=2e-3, atol=2e-4)
        print(f"checked M={M}, N={N}, K={K}")

    # 在目标 GPU 上可检查生成源码中的 MMA/WGMMA 指令。
    # print(kernel.get_kernel_source())
    # 测速前先编译并预热；本脚本不填入预设吞吐量。


if __name__ == "__main__":
    main()
