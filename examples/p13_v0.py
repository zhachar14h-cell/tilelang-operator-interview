# P13: Decode Attention
# Version: v0 · 单 query / 单 block
# Syntax checked only; GPU compilation and execution not verified.
import math
import torch
import tilelang
import tilelang.language as T


def make_decode_attention(S, D):
    # 教学范围：一个 batch、一个 head、一条 query。
    assert 1 <= S <= 256 and D in (32, 64)
    BS = 1 << (S - 1).bit_length()
    inv_sqrt_d = 1.0 / math.sqrt(D)

    @T.prim_func
    def main(
        Q: T.Tensor((D,), "float32"),
        K: T.Tensor((S, D), "float32"),
        V: T.Tensor((S, D), "float32"),
        O: T.Tensor((D,), "float32"),
    ):
        with T.Kernel(1, threads=128):
            qk = T.alloc_fragment((BS, D), "float32")
            scores = T.alloc_fragment((BS,), "float32")
            m = T.alloc_fragment((1,), "float32")
            denom = T.alloc_fragment((1,), "float32")
            weighted = T.alloc_fragment((BS, D), "float32")
            out = T.alloc_fragment((D,), "float32")

            # 每个有效 key 与唯一 query 做点积。
            for s, d in T.Parallel(BS, D):
                if s < S:
                    qk[s, d] = Q[d] * K[s, d]
                else:
                    qk[s, d] = 0.0
            T.reduce_sum(qk, scores, dim=1, clear=True)
            for s in T.Parallel(BS):
                if s < S:
                    scores[s] = scores[s] * inv_sqrt_d
                else:
                    scores[s] = -T.infinity("float32")

            # 稳定 softmax；padding 对分母贡献为零。
            T.reduce_max(scores, m, dim=0, clear=True)
            for s in T.Parallel(BS):
                scores[s] = T.exp(scores[s] - m[0])
            T.reduce_sum(scores, denom, dim=0, clear=True)
            for s in T.Parallel(BS):
                scores[s] = scores[s] / denom[0]

            # 对序列维求加权和，得到一个 D 维输出。
            for s, d in T.Parallel(BS, D):
                if s < S:
                    weighted[s, d] = scores[s] * V[s, d]
                else:
                    weighted[s, d] = 0.0
            T.reduce_sum(weighted, out, dim=0, clear=True)
            T.copy(out, O)

    return tilelang.compile(main, out_idx=None, target="cuda")


def main():
    torch.manual_seed(0)
    for S, D in [(1, 32), (127, 64), (256, 64)]:
        q = torch.randn(D, device="cuda")
        k = torch.randn(S, D, device="cuda")
        v = torch.randn(S, D, device="cuda")
        out = torch.empty(D, device="cuda")
        kernel = make_decode_attention(S, D)
        kernel(q, k, v, out)
        # FP64 reference 降低 reference 自身归约误差。
        score = (k.double() @ q.double()) / math.sqrt(D)
        ref = (torch.softmax(score, dim=0) @ v.double()).float()
        torch.testing.assert_close(out, ref, rtol=2e-4, atol=2e-5)
    print("Decode attention checks passed")


if __name__ == "__main__":
    main()
