# P07: Stable Row Softmax
# Version: v0 · 单行寄存器融合
# Syntax checked only; GPU compilation and execution not verified.
# Status: syntax checked only; not compiled or run on a GPU here.
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def make_softmax(M, N):
    assert M > 0 and 1 <= N <= 4096
    BLOCK = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"),
             Y: T.Tensor((M, N), "float32")):
        with T.Kernel(M, threads=128) as row:
            values = T.alloc_fragment((1, BLOCK), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            denominator = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(BLOCK):
                values[0, j] = T.if_then_else(
                    j < N, X[row, j], -T.infinity("float32"))
            T.reduce_max(values, maximum, dim=1)
            for j in T.Parallel(BLOCK):
                values[0, j] = T.exp(values[0, j] - maximum[0])
            T.reduce_sum(values, denominator, dim=1)
            for j in T.Parallel(BLOCK):
                if j < N:
                    Y[row, j] = values[0, j] / denominator[0]
    return main


def check(M, N):
    assert torch.cuda.is_available()
    x = (torch.randn((M, N), device="cuda") * 10).contiguous()
    x[0].fill_(1000.0)
    if M > 1:
        x[1].add_(1000.0)
    y = torch.empty_like(x)
    make_softmax(M, N)(x, y)
    torch.testing.assert_close(y, torch.softmax(x, -1),
                               rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(y.sum(-1), torch.ones(M, device="cuda"),
                               rtol=1e-5, atol=1e-5)
    print(f"passed: M={M}, N={N}")


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in (1, 127, 1024, 4095):
        check(32, n)
