# P18: Per-row INT8 Quantization
# Version: v0 · 逐行对称 INT8
# Syntax checked only; GPU compilation and execution not verified.
import torch
import tilelang
import tilelang.language as T


def make_row_int8_quant(M, N):
    assert M > 0 and 1 <= N <= 4096
    BN = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Q: T.Tensor((M, N), "int8"),
        Scale: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=128) as row:
            abs_x = T.alloc_fragment((BN,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            scale = T.alloc_fragment((1,), "float32")
            for col in T.Parallel(BN):
                if col < N:
                    abs_x[col] = T.abs(X[row, col])
                else:
                    abs_x[col] = 0.0
            T.reduce_max(abs_x, amax, dim=0, clear=True)
            for i in T.Parallel(1):
                scale[i] = T.if_then_else(
                    amax[i] == 0.0, 1.0, amax[i] / 127.0
                )
                Scale[row] = scale[i]

            for col in T.Parallel(BN):
                if col < N:
                    z = X[row, col] / scale[0]
                    floor_z = T.floor(z)
                    lo = T.cast(floor_z, "int32")
                    fraction = z - floor_z
                    # 最近整数；恰好位于中点时选择偶数。
                    rounded = T.if_then_else(
                        fraction < 0.5, lo,
                        T.if_then_else(
                            fraction > 0.5, lo + 1,
                            T.if_then_else(lo % 2 == 0, lo, lo + 1),
                        ),
                    )
                    clipped = T.max(-127, T.min(127, rounded))
                    Q[row, col] = T.cast(clipped, "int8")

    return tilelang.compile(main, out_idx=None, target="cuda")


def main():
    torch.manual_seed(0)
    M, N = 17, 513
    x = torch.randn(M, N, device="cuda") * 8.0
    x[0].zero_()
    # amax=127 => scale=1；同时检查正负中点的偶数舍入。
    x[1].zero_()
    x[1, :8] = torch.tensor(
        [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -127.0],
        device="cuda",
    )
    q = torch.empty(M, N, device="cuda", dtype=torch.int8)
    scale = torch.empty(M, device="cuda")
    make_row_int8_quant(M, N)(x, q, scale)

    amax = x.abs().amax(dim=1)
    ref_scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
    ref_q = torch.round(x / ref_scale[:, None]).clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    assert float(scale[0]) == 1.0
    print("INT8 codes, zero row and ties-to-even checked")


if __name__ == "__main__":
    main()
