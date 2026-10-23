# P06: Prefix Scan
# Version: V0 · 单行单块，双语义
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_row_scan(M, N, exclusive=False):
    assert M > 0 and 1 <= N <= 1024
    BLOCK = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=256) as row:
            values = T.alloc_shared((BLOCK,), "float32")
            for j in T.Parallel(BLOCK):
                if j < N:
                    values[j] = X[row, j]
                else:
                    values[j] = 0.0
            T.cumsum(src=values, dst=values, dim=0)
            for j in T.Parallel(BLOCK):
                if j < N:
                    if exclusive:
                        if j == 0:
                            Y[row, j] = 0.0
                        else:
                            Y[row, j] = values[j - 1]
                    else:
                        Y[row, j] = values[j]

    return tilelang.compile(main, target="cuda")


def main():
    torch.manual_seed(0)
    for M, N in [(17, 1), (37, 257), (64, 1024)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        inclusive = torch.cumsum(x, dim=1)
        for exclusive in [False, True]:
            y = torch.empty_like(x)
            kernel = make_row_scan(M, N, exclusive)
            kernel(x, y)
            if exclusive:
                ref = torch.cat(
                    [torch.zeros_like(x[:, :1]), inclusive[:, :-1]], dim=1
                )
            else:
                ref = inclusive
            torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-3)
    print("Inclusive / exclusive row-scan reference checks passed")


if __name__ == "__main__":
    main()
