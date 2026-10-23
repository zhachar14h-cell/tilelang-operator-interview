# P11: Naive Attention
# Version: v0 · 三个 TileLang kernel
# Syntax checked only; GPU compilation and execution not verified.
# Status: syntax checked only; not compiled or run on a GPU here.
# B=H=1, D=64, S divisible by 64, non-causal, FP16 Q/K/V.
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def make_qk(S, D=64):
    assert 0 < S <= 2048 and S % 64 == 0 and D == 64
    @T.prim_func
    def main(Q: T.Tensor((S, D), "float16"),
             K: T.Tensor((S, D), "float16"),
             Scores: T.Tensor((S, S), "float32")):
        with T.Kernel(S // 64, S // 64, threads=128) as (kc, qr):
            qs = T.alloc_shared((64, D), "float16")
            ks = T.alloc_shared((64, D), "float16")
            acc = T.alloc_fragment((64, 64), "float32")
            T.copy(Q[qr * 64, 0], qs)
            T.copy(K[kc * 64, 0], ks)
            T.clear(acc)
            T.gemm(qs, ks, acc, transpose_B=True)
            for i, j in T.Parallel(64, 64):
                acc[i, j] *= D ** -0.5
            T.copy(acc, Scores[qr * 64, kc * 64])
    return main


@tilelang.jit(target="cuda")
def make_probability(S):
    assert 0 < S <= 2048 and S % 64 == 0
    BLOCK = 1 << (S - 1).bit_length()
    @T.prim_func
    def main(Scores: T.Tensor((S, S), "float32"),
             P: T.Tensor((S, S), "float16")):
        with T.Kernel(S, threads=128) as row:
            v = T.alloc_fragment((1, BLOCK), "float32")
            m = T.alloc_fragment((1,), "float32")
            z = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(BLOCK):
                v[0, j] = T.if_then_else(
                    j < S, Scores[row, j], -T.infinity("float32"))
            T.reduce_max(v, m, dim=1)
            for j in T.Parallel(BLOCK):
                v[0, j] = T.exp(v[0, j] - m[0])
            T.reduce_sum(v, z, dim=1)
            for j in T.Parallel(BLOCK):
                if j < S:
                    P[row, j] = v[0, j] / z[0]
    return main


@tilelang.jit(target="cuda")
def make_pv(S, D=64):
    assert 0 < S <= 2048 and S % 64 == 0 and D == 64
    @T.prim_func
    def main(P: T.Tensor((S, S), "float16"),
             V: T.Tensor((S, D), "float16"),
             O: T.Tensor((S, D), "float16")):
        with T.Kernel(S // 64, threads=128) as qr:
            ps = T.alloc_shared((64, 64), "float16")
            vs = T.alloc_shared((64, D), "float16")
            acc = T.alloc_fragment((64, D), "float32")
            T.clear(acc)
            for tile in T.Pipelined(S // 64, num_stages=1):
                T.copy(P[qr * 64, tile * 64], ps)
                T.copy(V[tile * 64, 0], vs)
                T.gemm(ps, vs, acc)
            T.copy(acc, O[qr * 64, 0])
    return main


def check(S=256, D=64):
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability()[0] >= 8, "Use SM80+"
    q, k, v = [torch.randn((S, D), device="cuda", dtype=torch.float16)
               for _ in range(3)]
    scores = torch.empty((S, S), device="cuda", dtype=torch.float32)
    p = torch.empty((S, S), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    qk, sm, pv = make_qk(S, D), make_probability(S), make_pv(S, D)
    qk(q, k, scores)
    sm(scores, p)
    pv(p, v, out)
    ref_scores = q.float() @ k.float().T / (D ** 0.5)
    ref_p = torch.softmax(ref_scores, dim=-1)
    # Match the explicit FP16 probability boundary.
    ref_out = (ref_p.half().float() @ v.float()).half()
    torch.testing.assert_close(scores, ref_scores, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(p.float(), ref_p, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    print("three-stage attention passed")


if __name__ == "__main__":
    torch.manual_seed(0)
    check()
