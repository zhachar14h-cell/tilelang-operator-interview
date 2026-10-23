# P01: Bias + GELU 融合
# Version: V1 · 连续列 fragment
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_gelu_bias(M, N, BLOCK=1024):
    assert M > 0 and N % BLOCK == 0
    assert BLOCK % (256 * 4) == 0

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Bias: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(N // BLOCK, M, threads=256) as (bx, row):
            x = T.alloc_fragment((BLOCK,), "float32")
            b = T.alloc_fragment((BLOCK,), "float32")
            y = T.alloc_fragment((BLOCK,), "float32")
            col0 = bx * BLOCK
            T.copy(X[row, col0], x, coalesced_width=4)
            T.copy(Bias[col0], b, coalesced_width=4)
            for i in T.Parallel(BLOCK):
                v = x[i] + b[i]
                u = 0.7978845608 * (v + 0.044715 * v * v * v)
                y[i] = 0.5 * v * (1.0 + T.tanh(u))
            T.copy(y, Y[row, col0], coalesced_width=4)

    return tilelang.compile(main, target="cuda")


def main():
    torch.manual_seed(0)
    M = N = 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    kernel = make_gelu_bias(M, N)
    kernel(x, b, y)
    v = x + b
    ref = 0.5 * v * (1.0 + torch.tanh(
        0.7978845608 * (v + 0.044715 * v * v * v)))
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
    # 可检查 kernel.get_kernel_source() 中的访存与寄存器使用。
    print("GELU bias reference check passed")


if __name__ == "__main__":
    main()
