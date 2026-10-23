# TileLang 算子面试练习室

中文 TileLang 学习与面试练习站：**9 个模块、18 道题、21 份完整教学示例**。

- [在线练习站](https://tilelang-operator-interview-lab.quick-mug-8844.chatgpt.site)（公开访问，无需登录）
- [GitHub Pages](https://zhachar14h-cell.github.io/tilelang-operator-interview/)
- [下载教学代码包](docs/downloads/tilelang-practice.zip)

题目范围与练习形式参考 [CUDA 算子面试站](https://kkcocoa.github.io/cuda-operator-interview/)。教学文字和代码重新编写为 TileLang 版本，并链接官方资料。本项目为独立学习资料，不隶属于参考站或 TileLang 官方项目。

## 你可以怎样练

每题包含输入约束、公式、tile 划分、CUDA 概念对应、完整参考代码、填空与手写区、面试追问和检查清单。网站还支持搜索、模块筛选、学习进度、笔记、计时器与明暗主题。

Tile 映射实验台展示可选的数据分配方式；带宽计算器使用你提供的时间计算有效带宽，不包含虚构的 GPU 实测结果。

## 代码验证状态

**所有示例已通过 Python 语法检查，尚未完成 GPU 编译、正确性和性能验证。**

脚本包含需要在 GPU 上执行的 PyTorch 对照断言。教学实现有明确的形状、dtype 和数值范围限制，不等同于通用生产算子。例如部分 Attention 示例限定 B=H=1，FlashAttention 示例限定 D=64、非 causal 和 SM80+。

## 本地预览网站

```sh
python -m http.server 8000 --directory docs
```

在浏览器打开 `http://localhost:8000`。这是无需构建依赖的静态站；笔记与学习记录保存在当前浏览器，不上传服务器。

## 运行算子示例

先安装匹配设备与驱动的 CUDA 版 PyTorch 和 TileLang，具体支持范围见 [官方安装说明](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md)。然后运行：

```sh
python examples/p01_v0.py
```

测速前先完成正确性检查，将编译、内存分配和初始化放到计时外。[通用测速助手](docs/downloads/benchmark_helper.py) 提供 `measure(fn, useful_bytes)`；有效带宽不等同于硬件实测 DRAM 流量。

## 题目目录

| 编号 | 模块 | 题目 | 教学代码 |
| --- | --- | --- | --- |
| P01 | M0 | Bias + GELU 融合 | [v0](examples/p01_v0.py) / [v1](examples/p01_v1.py) |
| P02 | M0 | 矩阵转置 | [v0](examples/p02_v0.py) / [v1](examples/p02_v1.py) |
| P03 | M1 | 全局 Reduce Sum | [v0](examples/p03_v0.py) |
| P04 | M1 | 按行 Sum / Max 工具箱 | [v0](examples/p04_v0.py) |
| P05 | M1 | Max / Argmax | [v0](examples/p05_v0.py) |
| P06 | M2 | Prefix Scan | [v0](examples/p06_v0.py) |
| P07 | M3 | Stable Row Softmax | [v0](examples/p07_v0.py) |
| P08 | M3 | LayerNorm | [v0](examples/p08_v0.py) |
| P09 | M3 | RMSNorm + Residual | [v0](examples/p09_v0.py) / [v1](examples/p09_v1.py) |
| P10 | M4 | RoPE 双布局 | [v0](examples/p10_v0.py) |
| P11 | M5 | Naive Attention | [v0](examples/p11_v0.py) |
| P12 | M5 | Tiled FlashAttention | [v0](examples/p12_v0.py) |
| P13 | M5 | Decode Attention | [v0](examples/p13_v0.py) |
| P14 | M6 | GEMM | [v0](examples/p14_v0.py) |
| P15 | M6 | GEMV | [v0](examples/p15_v0.py) |
| P16 | M7 | TopK | [v0](examples/p16_v0.py) |
| P17 | M7 | Sort | [v0](examples/p17_v0.py) |
| P18 | M8 | Per-row INT8 Quantization | [v0](examples/p18_v0.py) |

## 维护教学内容

编辑根目录三份 `content-*.json` 后运行：

```sh
python prepare_content.py
```

该命令检查 18 题结构与 Python 语法，重新生成 `docs/curriculum.js`、`examples/`、练习 ZIP 和 `validation.json`。不要把语法检查结果解释为 GPU 测试通过。

## GitHub Pages

静态站位于 `docs/`，可使用 **Settings → Pages → Deploy from a branch → main → /docs** 发布。

项目不包含 Sites 的项目身份文件、访问凭据、本地笔记或浏览器学习记录。
