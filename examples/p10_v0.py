# P10: RoPE 双布局
# Version: v0 · 一份 factory，两种布局
# Syntax checked only; GPU compilation and execution not verified.
# Status: syntax checked only; not compiled or run on a GPU here.
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def make_rope(S, D, interleaved=True, BLOCK=256):
    assert S > 0 and D > 0 and D % 2 == 0
    HALF = D // 2
    STEP = 2 if interleaved else 1
    OFFSET = 1 if interleaved else HALF

    @T.prim_func
    def main(X: T.Tensor((S, D), "float32"),
             Cos: T.Tensor((S, HALF), "float32"),
             Sin: T.Tensor((S, HALF), "float32"),
             Y: T.Tensor((S, D), "float32")):
        with T.Kernel(T.ceildiv(S * HALF, BLOCK), threads=128) as bx:
            for lane in T.Parallel(BLOCK):
                pair = bx * BLOCK + lane
                if pair < S * HALF:
                    row = pair // HALF
                    j = pair % HALF
                    ia = j * STEP
                    ib = ia + OFFSET
                    a = X[row, ia]
                    b = X[row, ib]
                    c = Cos[row, j]
                    s = Sin[row, j]
                    Y[row, ia] = a * c - b * s
                    Y[row, ib] = a * s + b * c
    return main


def reference(x, cos, sin, interleaved):
    if interleaved:
        a, b = x[:, 0::2], x[:, 1::2]
        return torch.stack((a * cos - b * sin,
                            a * sin + b * cos), dim=-1).flatten(-2)
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)


def check(S=17, D=64):
    x = torch.randn((S, D), device="cuda", dtype=torch.float32)
    position = torch.arange(S, device="cuda", dtype=torch.float32)
    j = torch.arange(D // 2, device="cuda", dtype=torch.float32)
    inverse_frequency = torch.pow(10000.0, -2.0 * j / D)
    angle = position[:, None] * inverse_frequency[None, :]
    cos, sin = angle.cos().contiguous(), angle.sin().contiguous()
    for interleaved in (True, False):
        y = torch.empty_like(x)
        make_rope(S, D, interleaved)(x, cos, sin, y)
        ref = reference(x, cos, sin, interleaved)
        torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(y.square().sum(-1), x.square().sum(-1),
                                   rtol=1e-5, atol=1e-5)
        print(f"passed: interleaved={interleaved}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    check()
