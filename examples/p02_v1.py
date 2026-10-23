# P02: 矩阵转置
# Version: V1 · 共享内存重排
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
            src = T.alloc_shared((TILE, TILE), "float32")
            dst = T.alloc_shared((TILE, TILE), "float32")
            # 所有 src 元素都定义；边缘位置补零。
            for i, j in T.Parallel(TILE, TILE):
                row = by * TILE + i
                col = bx * TILE + j
                if row < M and col < N:
                    src[i, j] = X[row, col]
                else:
                    src[i, j] = 0.0
            # dst[j,i] = src[i,j]，默认流程负责共享内存同步。
            T.transpose(src, dst)
            for i, j in T.Parallel(TILE, TILE):
                out_row = bx * TILE + i
                out_col = by * TILE + j
                if out_row < N and out_col < M:
                    Y[out_row, out_col] = dst[i, j]

    return tilelang.compile(main, target="cuda")


def main():
    for M, N in [(37, 65), (257, 129), (1024, 2048)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        y = torch.empty(N, M, device="cuda", dtype=torch.float32)
        kernel = make_transpose(M, N)
        kernel(x, y)
        torch.testing.assert_close(y, x.t().contiguous(), rtol=0, atol=0)
    print("Shared transpose reference checks passed")


if __name__ == "__main__":
    main()
