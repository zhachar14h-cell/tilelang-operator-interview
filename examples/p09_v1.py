# P09: RMSNorm + Residual
# Version: v1 · 融合 Residual Add
# Syntax checked only; GPU compilation and execution not verified.
# Status: syntax checked only; not compiled or run on a GPU here.
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def make_residual_rmsnorm(M, N, EPS=1e-6):
    assert M > 0 and 1 <= N <= 4096 and EPS > 0
    BLOCK = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"),
             R: T.Tensor((M, N), "float32"),
             W: T.Tensor((N,), "float32"),
             Y: T.Tensor((M, N), "float32")):
        with T.Kernel(M, threads=128) as row:
            combined = T.alloc_fragment((1, BLOCK), "float32")
            squares = T.alloc_fragment((1, BLOCK), "float32")
            square_sum = T.alloc_fragment((1,), "float32")
            inv_rms = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(BLOCK):
                combined[0, j] = T.if_then_else(
                    j < N, X[row, j] + R[row, j], 0.0)
                squares[0, j] = combined[0, j] * combined[0, j]
            T.reduce_sum(squares, square_sum, dim=1)
            for r in T.Parallel(1):
                inv_rms[r] = T.rsqrt(square_sum[r] / N + EPS)
            for j in T.Parallel(BLOCK):
                if j < N:
                    Y[row, j] = combined[0, j] * inv_rms[0] * W[j]
    return main


def check(M=32, N=1023):
    eps = 1e-6
    x = torch.randn((M, N), device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(N, device="cuda")
    y = torch.empty_like(x)
    make_residual_rmsnorm(M, N, eps)(x, residual, weight, y)
    u = x + residual
    ref = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)
    ref = ref * weight
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-5)
    print("residual RMSNorm passed")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    check()
