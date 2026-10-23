# TileLang 算子练习包

包含 18 道题的完整教学示例。题目范围参考：
https://kkcocoa.github.io/cuda-operator-interview/
教学文字和代码重新编写，采用 TileLang 的数据分块与计算原语。

## 验证状态

所有 Python 示例通过语法检查。尚未完成 GPU 编译、正确性和性能验证。
脚本内包含 PyTorch 参考计算和断言；这些断言需在你的环境执行。
阅读各题的输入限制后再运行，不要把教学实现当成通用生产算子。

## 运行

1. 准备兼容的 CUDA 驱动、工具链、CUDA 版 PyTorch 和 TileLang。
   安装文档：https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md
2. 在相应环境运行：python p01_v0.py
3. v0/v1 是该题参考版本，具体语义与输入边界见代码和学习站。
4. benchmark_helper.py 提供 measure(fn, useful_bytes)，用于预编译、
   预分配和已完成正确性检查的 callable。不要计入初始化或编译时间。

## 练习建议

先解释每个 block 负责的数据，再独立手写，最后对照实现和检查生成代码。
记录 GPU / CUDA / PyTorch / TileLang 版本、shape、dtype、配置与容差。
性能数字必须来自实际设备；单纯估计 FLOPs 不能保证某个算子受带宽限制。
