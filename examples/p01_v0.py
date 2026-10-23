# P01: Bias + GELU 融合
# Version: V0 · 线性 tile，显式尾部保护
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_gelu_bias(M, N, BLOCK=256):
    assert M > 0 and N > 0

    @T.prim_func
    def main(
        X: T.Tensor((M * N,), "float32"),
        Bias: T.Tensor((N,), "float32"),
        Y: T.Tensor((M * N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M * N, BLOCK), threads=256) as bx:
            for i in T.Parallel(BLOCK):
                idx = bx * BLOCK + i
                if idx < M * N:
                    v = X[idx] + Bias[idx % N]
                    u = 0.7978845608 * (v + 0.044715 * v * v * v)
                    Y[idx] = 0.5 * v * (1.0 + T.tanh(u))

    # 不设置 out_idx：调用者显式传入输出缓冲区。
    return tilelang.compile(main, target="cuda")


def main():
    torch.manual_seed(0)
    # 同时覆盖题目尺寸与非整除尾部。
    for M, N in [(4096, 4096), (7, 513)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        kernel = make_gelu_bias(M, N)
        kernel(x.view(-1), bias, y.view(-1))
        v = x + bias
        ref = 0.5 * v * (1.0 + torch.tanh(
            0.7978845608 * (v + 0.044715 * v * v * v)))
        torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
    print("GELU bias reference checks passed")


if __name__ == "__main__":
    main()
