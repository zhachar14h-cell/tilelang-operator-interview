# P08: LayerNorm
# Version: v0 · 均值 + 中心方差
# Syntax checked only; GPU compilation and execution not verified.
# Status: syntax checked only; not compiled or run on a GPU here.
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def make_layernorm(M, N, EPS=1e-5):
    assert M > 0 and 1 <= N <= 4096 and EPS > 0
    BLOCK = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"),
             G: T.Tensor((N,), "float32"),
             B: T.Tensor((N,), "float32"),
             Y: T.Tensor((M, N), "float32")):
        with T.Kernel(M, threads=128) as row:
            values = T.alloc_fragment((1, BLOCK), "float32")
            centered_sq = T.alloc_fragment((1, BLOCK), "float32")
            total = T.alloc_fragment((1,), "float32")
            mean = T.alloc_fragment((1,), "float32")
            variance = T.alloc_fragment((1,), "float32")
            invstd = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(BLOCK):
                values[0, j] = T.if_then_else(j < N, X[row, j], 0.0)
            T.reduce_sum(values, total, dim=1)
            for r in T.Parallel(1):
                mean[r] = total[r] / N
            # Second statistical pass: centered squared differences.
            for j in T.Parallel(BLOCK):
                centered_sq[0, j] = T.if_then_else(
                    j < N, (values[0, j] - mean[0]) * (values[0, j] - mean[0]), 0.0)
            T.reduce_sum(centered_sq, variance, dim=1)
            for r in T.Parallel(1):
                invstd[r] = T.rsqrt(variance[r] / N + EPS)
            for j in T.Parallel(BLOCK):
                if j < N:
                    Y[row, j] = ((values[0, j] - mean[0])
                                 * invstd[0] * G[j] + B[j])
    return main


def check(N, offset):
    M, eps = 32, 1e-5
    x = (torch.randn((M, N), device="cuda") + offset).contiguous()
    x[0].fill_(offset)
    gamma = torch.randn(N, device="cuda")
    beta = torch.randn(N, device="cuda")
    y = torch.empty_like(x)
    make_layernorm(M, N, eps)(x, gamma, beta, y)
    ref = F.layer_norm(x, (N,), gamma, beta, eps)
    torch.testing.assert_close(y, ref, rtol=3e-4, atol=3e-4)
    print(f"passed: N={N}, offset={offset}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    for n in (1, 255, 1024, 4095):
        check(n, 0.0)
        check(n, 100.0)
