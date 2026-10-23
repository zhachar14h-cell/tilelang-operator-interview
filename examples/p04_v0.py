# P04: 按行 Sum / Max 工具箱
# Version: V0 · 双归约基线
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_row_stats(M, N):
    assert M > 0 and 1 <= N <= 4096
    BLOCK = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Sum: T.Tensor((M,), "float32"),
        Max: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=256) as row:
            xs = T.alloc_fragment((1, BLOCK), "float32")
            xm = T.alloc_fragment((1, BLOCK), "float32")
            s = T.alloc_fragment((1,), "float32")
            m = T.alloc_fragment((1,), "float32")
            for i, j in T.Parallel(1, BLOCK):
                if j < N:
                    xs[i, j] = X[row, j]
                    xm[i, j] = X[row, j]
                else:
                    xs[i, j] = 0.0
                    xm[i, j] = -T.infinity("float32")
            T.reduce_sum(xs, s, dim=1)
            T.reduce_max(xm, m, dim=1)
            for i in T.Parallel(1):
                Sum[row] = s[i]
                Max[row] = m[i]

    return tilelang.compile(main, target="cuda")


def main():
    torch.manual_seed(0)
    for M, N in [(257, 1001), (17, 4096), (5, 1)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        # 确保错误地以 0 填充 max 时会被发现。
        x[0] = -x[0].abs() - 1.0
        sums = torch.empty(M, device="cuda", dtype=torch.float32)
        maxima = torch.empty_like(sums)
        kernel = make_row_stats(M, N)
        kernel(x, sums, maxima)
        torch.testing.assert_close(sums, x.sum(dim=1), rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(maxima, x.amax(dim=1), rtol=0, atol=0)
    print("Row sum/max reference checks passed")


if __name__ == "__main__":
    main()
