# P17: Sort
# Version: v0 · bitonic 双缓冲基础版
# Syntax checked only; GPU compilation and execution not verified.
import torch
import tilelang
import tilelang.language as T


def make_row_sort(M, N):
    # 仅升序值排序；N 为 2 的幂，输入不含 NaN/Inf。
    assert M > 0 and 1 <= N <= 1024 and (N & (N - 1)) == 0
    LOG_N = N.bit_length() - 1

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=128) as row:
            a = T.alloc_shared((N,), "float32")
            b = T.alloc_shared((N,), "float32")
            T.copy(X[row, 0], a)
            T.sync_threads()

            for level in T.serial(1, LOG_N + 1):
                width = (T.int32(1) << level)
                for phase in T.serial(level):
                    stride = (T.int32(1) << (level - 1 - phase))
                    # 这一轮 a 是只读快照，b 是每个位置唯一的写入目标。
                    for i in T.Parallel(N):
                        partner = T.if_then_else(
                            (i // stride) % 2 == 0,
                            i + stride,
                            i - stride,
                        )
                        # 等价于 bitonic 的方向位和 pair 内高低位比较。
                        keep_min = (i // width) % 2 == (i // stride) % 2
                        if keep_min:
                            b[i] = T.min(a[i], a[partner])
                        else:
                            b[i] = T.max(a[i], a[partner])

                    # 全部 b 完成之后才覆盖 a。
                    T.sync_threads()
                    for i in T.Parallel(N):
                        a[i] = b[i]
                    # 全部 a 完成之后才进入下一轮比较。
                    T.sync_threads()

            T.copy(a, Y[row, 0])

    return tilelang.compile(main, out_idx=None, target="cuda")


def main():
    torch.manual_seed(0)
    for M, N in [(3, 1), (7, 32), (17, 512)]:
        # 重复值和负数同样参与排序；不承诺元素稳定性。
        x = torch.randint(-20, 21, (M, N), device="cuda").float()
        y = torch.empty_like(x)
        make_row_sort(M, N)(x, y)
        ref = torch.sort(x, dim=1).values
        torch.testing.assert_close(y, ref, rtol=0, atol=0)
        if N > 1:
            assert bool(torch.all(y[:, 1:] >= y[:, :-1]))
        print(f"checked M={M}, N={N}")


if __name__ == "__main__":
    main()
