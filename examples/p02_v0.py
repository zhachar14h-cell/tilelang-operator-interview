# P02: 矩阵转置
# Version: V0 · 直接索引
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_transpose(M, N, TILE=32):
    assert M > 0 and N > 0

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((N, M), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(N, TILE), T.ceildiv(M, TILE), threads=256
        ) as (bx, by):
            for i, j in T.Parallel(TILE, TILE):
                row = by * TILE + i
                col = bx * TILE + j
                if row < M and col < N:
                    Y[col, row] = X[row, col]

    return tilelang.compile(main, target="cuda")


def main():
    # 矩形、边缘和整齐尺寸都进入同一套验证流程。
    for M, N in [(37, 65), (257, 129), (1024, 2048)]:
        x = torch.arange(M * N, device="cuda", dtype=torch.float32)
        x = x.reshape(M, N)
        y = torch.empty(N, M, device="cuda", dtype=torch.float32)
        kernel = make_transpose(M, N)
        kernel(x, y)
        ref = x.t().contiguous()
        torch.testing.assert_close(y, ref, rtol=0, atol=0)
        assert y.is_contiguous()
    print("Transpose reference checks passed")


if __name__ == "__main__":
    main()
