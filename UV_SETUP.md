# uv 环境管理说明（yoloe）

本仓库的环境已从「pip + 手工 venv」切换到 **uv 项目模式**：`pyproject.toml` 描述依赖，`uv.lock` 锁定精确版本，`uv sync` 把 `.venv` 还原成与锁文件完全一致的状态。

## 常用命令

```powershell
cd D:\a i人物动作时序识别\yoloe

uv sync                 # 按 uv.lock 同步环境（精确同步：会卸载锁文件外的包）
uv sync --check         # 只校验环境是否与锁文件一致，不改动（CI 用）
uv sync --inexact       # 保留锁文件之外的多余包
uv run python xxx.py    # 在项目环境里跑（先自动 sync）
uv add <包名>            # 新增依赖并更新锁文件
uv remove <包名>         # 移除依赖
uv lock                 # 只重新解析依赖，不动环境
uv lock --upgrade-package torch   # 只升级指定包
uv tree --invert --package <包名>  # 查某个包被谁依赖
```

> ⚠️ `uv sync` 是**精确同步**：锁文件里没有的包会被卸载。要「只加不减」请用 `uv add` / `uv pip install`，或给 `uv sync` 加 `--inexact`。

## 本项目的关键配置（`pyproject.toml` 末尾）

| 配置 | 作用 |
|---|---|
| `[tool.uv] environments` | 只解析 `win32 + 非 aarch64 + Python 3.13` 这一个分支。写 `platform_machine == 'x86_64'` 会永远匹配不上（uv 对 Windows 轮子求值用的是 `AMD64`），必须用 `!= 'aarch64'` |
| `[tool.uv] constraint-dependencies` | 把已验证可跑的版本钉死（torch / numpy / opencv / pydantic / huggingface-hub 等），防止 sync 时被动升级 |
| `[tool.uv.sources]` | `lvis` / `mobileclip` / `clip` 指向 `third_party/` 本地可编辑安装；`torch` / `torchvision` 指向 pytorch cu126 索引 |
| `[[tool.uv.index]] pytorch` | `https://download.pytorch.org/whl/cu126`，`explicit = true` 表示只有 torch/torchvision 从这里取 |
| `[tool.uv.extra-build-dependencies]` | `lvis` 的 setup.py 里有 `import pip`；`third_party/CLIP` 的 setup.py 用了 `pkg_resources`（新版 setuptools 已移除），构建环境需补 `setuptools<81` |

`.python-version` 固定为 `3.13`，防止 uv 选到 3.14 导致 torch 无对应轮子。

## 新增/删除依赖的正确姿势

```powershell
uv add pycocotools==2.0.10     # 写进 pyproject + 更新 uv.lock
uv remove somepkg               # 从 pyproject 和锁文件移除
uv sync                         # 让 .venv 跟上
```

不要直接 `uv pip install` 到环境里——那样装的东西不在锁文件里，下次 `uv sync` 会被卸载。

## 踩过的坑

1. **torch 必须带 `+cu126`**：cu126 索引上只有 `2.14.1+cu126` 这种局部版本，没有裸 `2.14.1`。约束写 `torch==2.14.1` 会解析失败；写 `torch==2.14.1+cu126` 才对。
2. **`platform_machine` 陷阱**：见上表，`!= 'aarch64'` 才是正确写法。
3. **上游 extras 冲突**：ultralytics 的 `export` extra 里有 `numpy==1.23.5; platform_machine == 'aarch64'` 这类分支，限制 `environments` 后才不会互相打架。
4. **Python 3.13 移除标准库 `audioop`**：`pydub` → `gradio` 链路会崩，装 `audioop-lts` 解决。
5. **`gradio 4.42` 与 `huggingface_hub` 不兼容**：`HfFolder` 在 hub 0.26 已删除，需要 `huggingface-hub==0.25.2`（同时会移除 `hf-xet`）。
6. **本地可编辑安装必须在仓库根目录执行**，相对路径 `third_party/...` 才对。
7. `pip freeze` 在本环境会崩（venv 路径编码问题），用 `uv pip list` / `uv pip compile` 代替。

## 验收

```powershell
uv run python ..\demo\check_env.py          # 逐项检查关键包 + CUDA
uv run python ..\demo\track_demo.py --max-frames 40 --no-save --out-dir ..\demo\out\uv_verify
```

`demo/check_env.py` 会打印 torch/CUDA 状态、关键包的导入结果，以及 `ultralytics` 是否指向仓库内的可编辑副本。
