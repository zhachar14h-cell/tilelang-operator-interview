# P16: TopK
# Version: v0 · K 次确定性 argmax
# Syntax checked only; GPU compilation and execution not verified.
import torch
import tilelang
import tilelang.language as T


def make_topk(M, N, K):
    # 有限 FP32 输入，小 K；相同值优先选择原始下标较小者。
    assert M > 0 and 1 <= N <= 2048
    assert 1 <= K <= min(8, N)
    BN = 1 << (N - 1).bit_length()

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Values: T.Tensor((M, K), "float32"),
        Indices: T.Tensor((M, K), "int32"),
    ):
        with T.Kernel(M, threads=128) as row:
            work = T.alloc_fragment((BN,), "float32")
            candidates = T.alloc_fragment((BN,), "int32")
            best = T.alloc_fragment((1,), "float32")
            first = T.alloc_fragment((1,), "int32")

            for col in T.Parallel(BN):
                if col < N:
                    work[col] = X[row, col]
                else:
                    work[col] = -T.infinity("float32")

            for rank in T.serial(K):
                T.reduce_max(work, best, dim=0, clear=True)
                # 先归约值，再在等于最大值的位置中选最小下标。
                for col in T.Parallel(BN):
                    candidates[col] = T.if_then_else(
                        work[col] == best[0], col, BN
                    )
                T.reduce_min(candidates, first, dim=0, clear=True)
                for i in T.Parallel(1):
                    Values[row, rank] = best[i]
                    Indices[row, rank] = first[i]
                # 只删除被选中的位置，重复值可在下一轮继续入选。
                for col in T.Parallel(BN):
                    if col == first[0]:
                        work[col] = -T.infinity("float32")

    return tilelang.compile(main, out_idx=None, target="cuda")


def main():
    torch.manual_seed(0)
    M, N, K = 17, 513, 8
    # 有意生成大量 ties，并增加整行相等的样例。
    x = torch.randint(-8, 9, (M, N), device="cuda").float()
    x[0].fill_(3.0)
    values = torch.empty(M, K, device="cuda")
    indices = torch.empty(M, K, device="cuda", dtype=torch.int32)
    make_topk(M, N, K)(x, values, indices)

    ref_idx = torch.argsort(x, dim=1, descending=True, stable=True)[:, :K]
    ref_values = torch.gather(x, 1, ref_idx)
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_idx.int(), rtol=0, atol=0)
    print("TopK values, indices and tie rule checked")


if __name__ == "__main__":
    main()
