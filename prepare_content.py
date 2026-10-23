"""Validate original curriculum, generate static data and downloadable scripts."""

import ast
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "docs"
REQUIRED = (
    "id module title subtitle difficulty minutes tags statement formula inputs "
    "outputs constraints approach mapping versions pitfalls questions checklist "
    "blanks references"
).split()
EXPECTED_MODULES = ["M0", "M0", "M1", "M1", "M1", "M2", "M3", "M3", "M3", "M4", "M5", "M5", "M5", "M6", "M6", "M7", "M7", "M8"]


def main():
    lessons = []
    for name in ("content-memory.json", "content-norm-attention.json", "content-advanced.json"):
        lessons.extend(json.loads((ROOT / name).read_text(encoding="utf-8")))
    lessons.sort(key=lambda x: x["id"])
    assert [p["id"] for p in lessons] == [f"P{i:02}" for i in range(1, 19)], "Expected P01-P18 exactly once"
    script_count = 0
    examples = ROOT / "examples"
    examples.mkdir(exist_ok=True)
    for i, lesson in enumerate(lessons):
        assert all(k in lesson for k in REQUIRED), lesson["id"]
        assert lesson["module"] == EXPECTED_MODULES[i], lesson["id"]
        assert lesson["difficulty"] in ("入门", "进阶", "挑战")
        assert len(lesson["questions"]) >= 3
        assert len(lesson["checklist"]) >= 3
        assert len(lesson["blanks"]) >= 2
        assert lesson["versions"] and lesson["constraints"]
        for ref in lesson["references"]:
            assert ref["url"].startswith("https://"), ref
        for j, version in enumerate(lesson["versions"]):
            ast.parse(version["code"], filename=f'{lesson["id"]}_v{j}.py')
            script_count += 1

    (DIST / "curriculum.js").write_text(
        "window.CURRICULUM = " + json.dumps(lessons, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    downloads = DIST / "downloads"
    downloads.mkdir(exist_ok=True)
    ast.parse((downloads / "benchmark_helper.py").read_text(encoding="utf-8"))
    readme = """# TileLang 算子练习包

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
"""
    (downloads / "README.md").write_text(readme, encoding="utf-8")
    with zipfile.ZipFile(downloads / "tilelang-practice.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", readme)
        archive.write(downloads / "benchmark_helper.py", "benchmark_helper.py")
        for lesson in lessons:
            for j, version in enumerate(lesson["versions"]):
                name = f'{lesson["id"].lower()}_v{j}.py'
                preamble = f'# {lesson["id"]}: {lesson["title"]}\n# Version: {version["name"]}\n# Syntax checked only; GPU compilation and execution not verified.\n'
                archive.writestr(name, preamble + version["code"])
                (examples / name).write_text(preamble + version["code"], encoding="utf-8")

    report = {
        "lessons": len(lessons), "modules": len(set(p["module"] for p in lessons)),
        "scripts": script_count, "python_syntax": "passed",
        "gpu_compilation": "not_run", "gpu_correctness": "not_run",
        "gpu_performance": "not_run",
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
