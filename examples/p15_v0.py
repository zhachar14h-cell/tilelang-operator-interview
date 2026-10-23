# P15: GEMV
# Version: v0 · 每行一个 block
# Syntax checked only; GPU compilation and execution not verified.
import torch
import tilelang
import tilelang.language as T


def make_gemv(M, N):
    # 一个 block 负责一行；完整归约列维。
    assert M > 0 and 1 <= N <= 4096
    BN = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        X: T.Tensor((N,), "float32"),
        Y: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=128) as row:
            products = T.alloc_fragment((BN,), "float32")
            total = T.alloc_fragment((1,), "float32")

            # row-major A 的连续列由并行循环处理。
            # padding 必须填 sum 的单位元零。
            for col in T.Parallel(BN):
                if col < N:
                    products[col] = A[row, col] * X[col]
                else:
                    products[col] = 0.0
            T.reduce_sum(products, total, dim=0, clear=True)
            for i in T.Parallel(1):
                Y[row] = total[i]

    return tilelang.compile(main, out_idx=None, target="cuda")


def main():
    torch.manual_seed(0)
    for M, N in [(1, 1), (37, 1000), (128, 4096)]:
        a = torch.randn(M, N, device="cuda")
        x = torch.randn(N, device="cuda")
        y = torch.empty(M, device="cuda")
        kernel = make_gemv(M, N)
        kernel(a, x, y)

        # 用 FP64 点积检查 FP32 归约，允许合理的加法次序误差。
        ref = (a.double() @ x.double()).float()
        torch.testing.assert_close(y, ref, rtol=1e-4, atol=3e-4)
        assert y.shape == (M,)
        assert y.dtype == torch.float32
        print(f"checked M={M}, N={N}")

    # 若 M 很小而 N 很长，再设计多 block 的分段归约。
    # 这份基础版本无需全局原子操作或额外归约 kernel。


if __name__ == "__main__":
    main()
