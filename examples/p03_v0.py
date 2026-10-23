# P03: 全局 Reduce Sum
# Version: V0 · 完整两阶段调用
# Syntax checked only; GPU compilation and execution not verified.
# Python 语法已检查；未在 GPU 上验证。
import torch
import tilelang
import tilelang.language as T


def make_block_sum(N, BLOCK=1024):
    assert N > 0
    P = (N + BLOCK - 1) // BLOCK

    @T.prim_func
    def main(
        X: T.Tensor((N,), "float32"),
        Partial: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(P, threads=256) as bx:
            values = T.alloc_fragment((1, BLOCK), "float32")
            total = T.alloc_fragment((1,), "float32")
            for i, j in T.Parallel(1, BLOCK):
                idx = bx * BLOCK + j
                if idx < N:
                    values[i, j] = X[idx]
                else:
                    values[i, j] = 0.0
            T.reduce_sum(values, total, dim=1)
            for i in T.Parallel(1):
                Partial[bx] = total[i]

    return tilelang.compile(main, target="cuda")


def prepare_two_stage(N, device, BLOCK=1024):
    assert 1 <= N <= BLOCK * BLOCK
    P = (N + BLOCK - 1) // BLOCK
    partial = torch.empty(P, device=device, dtype=torch.float32)
    y = torch.empty(1, device=device, dtype=torch.float32)
    stage1 = make_block_sum(N, BLOCK)
    stage2 = make_block_sum(P, BLOCK)
    # 返回已编译 kernel 与已分配工作区，方便放到计时外。
    return stage1, stage2, partial, y


def main():
    torch.manual_seed(0)
    for N in [1, 1025, 1_000_003]:
        x = torch.randn(N, device="cuda", dtype=torch.float32)
        stage1, stage2, partial, y = prepare_two_stage(N, x.device)
        # 同一执行流按顺序启动；第二阶段消费第一阶段结果。
        stage1(x, partial)
        stage2(partial, y)
        ref = x.sum().reshape(1)
        torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-2)
    print("Two-stage reduce-sum reference checks passed")


if __name__ == "__main__":
    main()
