# P05: Max / Argmax
# Version: V0 · 两遍块内归约
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_row_argmax(M, N):
    assert M > 0 and 1 <= N <= 4096
    BLOCK = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        V: T.Tensor((M,), "float32"),
        I: T.Tensor((M,), "int32"),
    ):
        with T.Kernel(M, threads=256) as row:
            x = T.alloc_fragment((1, BLOCK), "float32")
            vmax = T.alloc_fragment((1,), "float32")
            candidate = T.alloc_fragment((1, BLOCK), "int32")
            first = T.alloc_fragment((1,), "int32")
            for i, j in T.Parallel(1, BLOCK):
                if j < N:
                    x[i, j] = X[row, j]
                else:
                    x[i, j] = -T.infinity("float32")
            T.reduce_max(x, vmax, dim=1)
            for i, j in T.Parallel(1, BLOCK):
                candidate[i, j] = T.if_then_else(
                    (j < N) & (x[i, j] == vmax[i]),
                    j, 2147483647,
                )
            T.reduce_min(candidate, first, dim=1)
            for i in T.Parallel(1):
                V[row] = vmax[i]
                I[row] = first[i]

    return tilelang.compile(main, target="cuda")


def main():
    torch.manual_seed(0)
    M, N = 257, 1001
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    x[0] = -1.0
    x[0, 5] = 10.0
    x[0, 17] = 10.0  # 平值必须选择列 5。
    x[1] = -3.0      # 整行平值必须选择列 0。
    v = torch.empty(M, device="cuda", dtype=torch.float32)
    idx = torch.empty(M, device="cuda", dtype=torch.int32)
    kernel = make_row_argmax(M, N)
    kernel(x, v, idx)
    ref_v, ref_i = x.max(dim=1)
    torch.testing.assert_close(v, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(idx, ref_i.to(torch.int32), rtol=0, atol=0)
    print("Max / first-argmax reference checks passed")


if __name__ == "__main__":
    main()
