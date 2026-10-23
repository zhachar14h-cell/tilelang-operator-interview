# P12: Tiled FlashAttention
# Version: v0 · 非 causal Online Softmax
# Syntax checked only; GPU compilation and execution not verified.
# Status: syntax checked only; not compiled or run on a GPU here.
# B=H=1, D=64, S divisible by 64, non-causal, NVIDIA SM80+.
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def make_flash_attention(S, D=64):
    assert S > 0 and S % 64 == 0 and D == 64
    BM, BN = 64, 64
    SCALE = D ** -0.5

    @T.prim_func
    def main(Q: T.Tensor((S, D), "float16"),
             K: T.Tensor((S, D), "float16"),
             V: T.Tensor((S, D), "float16"),
             O: T.Tensor((S, D), "float16")):
        with T.Kernel(S // BM, threads=128) as qb:
            qs = T.alloc_shared((BM, D), "float16")
            ks = T.alloc_shared((BN, D), "float16")
            vs = T.alloc_shared((BN, D), "float16")
            ps = T.alloc_shared((BM, BN), "float16")
            score = T.alloc_fragment((BM, BN), "float32")
            out = T.alloc_fragment((BM, D), "float32")
            running_m = T.alloc_fragment((BM,), "float32")
            running_l = T.alloc_fragment((BM,), "float32")
            merged_m = T.alloc_fragment((BM,), "float32")
            tile_l = T.alloc_fragment((BM,), "float32")
            alpha = T.alloc_fragment((BM,), "float32")
            T.copy(Q[qb * BM, 0], qs)
            T.clear(out)
            T.clear(running_l)
            T.fill(running_m, -T.infinity("float32"))
            # Serial loop exposes dependencies of the online state.
            for kv in T.serial(S // BN):
                T.copy(K[kv * BN, 0], ks)
                T.copy(V[kv * BN, 0], vs)
                T.clear(score)
                T.gemm(qs, ks, score, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BM, BN):
                    score[i, j] *= SCALE
                T.reduce_max(score, merged_m, dim=1)
                for i in T.Parallel(BM):
                    merged_m[i] = T.max(running_m[i], merged_m[i])
                    alpha[i] = T.exp(running_m[i] - merged_m[i])
                for i, j in T.Parallel(BM, BN):
                    score[i, j] = T.exp(score[i, j] - merged_m[i])
                T.reduce_sum(score, tile_l, dim=1)
                for i in T.Parallel(BM):
                    running_l[i] = alpha[i] * running_l[i] + tile_l[i]
                    running_m[i] = merged_m[i]
                # Rescale the old numerator BEFORE adding this tile.
                for i, d in T.Parallel(BM, D):
                    out[i, d] *= alpha[i]
                T.copy(score, ps)  # FP32 weights -> FP16 GEMM input.
                T.gemm(ps, vs, out, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(BM, D):
                out[i, d] /= running_l[i]
            T.copy(out, O[qb * BM, 0])
    return main


def check(S=256, D=64):
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability()[0] >= 8, "Use SM80+"
    q, k, v = [torch.randn((S, D), device="cuda", dtype=torch.float16)
               for _ in range(3)]
    out = torch.empty_like(q)
    make_flash_attention(S, D)(q, k, v, out)
    logits = (q.float() @ k.float().T) / (D ** 0.5)
    ref = (torch.softmax(logits, -1) @ v.float()).half()
    # FP16 unnormalized tile weights are an approximation boundary.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    assert torch.isfinite(out).all()
    print("online attention passed")


if __name__ == "__main__":
    torch.manual_seed(0)
    check(64)
    check(256)
