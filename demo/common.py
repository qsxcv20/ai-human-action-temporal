"""
common.py — demo 各脚本共用的路径推导、cwd 切换与中文打印辅助

为什么单独抽一个 common
----------------------
1) YOLOE 的文本编码器用**相对路径**加载 mobileclip_blt.pt
   （ultralytics/nn/text_model.py:51），进程 cwd 必须是 yoloe/ 仓库根，
   否则一调用 get_text_pe/set_classes 就崩。以前靠文档里写死「cd yoloe」
   来保证，既脆弱也和实际运行目录脱节。
2) 仓库根以前用 parent.parent / "yoloe" 硬推导，等于假设 demo/ 与 yoloe/
   一定平级。这里统一改成基于 __file__ 推导，和进程 cwd 无关。
3) 相对路径参数（--source / --out-dir / --tracker）必须在**切 cwd 之前**
   就按用户当时的工作目录解析成绝对路径，否则切完目录就找不到文件了。
"""

import os
import unicodedata
from contextlib import contextmanager
from pathlib import Path

# 路径一律基于 __file__ 推导，与进程当前工作目录无关
DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parent / "yoloe"

PRETRAIN_DIR = REPO_ROOT / "pretrain"
DEFAULT_CKPT_NAME = "yoloe-11s-seg.pt"
DEFAULT_OUT = DEMO_ROOT / "out"

# ultralytics 自带的跟踪器配置名，原样传给 model.track() 即可
BUILTIN_TRACKERS = ("bytetrack.yaml", "botsort.yaml")


# ------------------------------------------------------------------ cwd 切换
def ensure_repo_cwd():
    """把进程 cwd 切到 yoloe 仓库根，返回切换前的 cwd（交给 restore_cwd 还原）。"""
    if not REPO_ROOT.is_dir():
        raise FileNotFoundError(f"找不到 yoloe 仓库根: {REPO_ROOT}")
    old = Path.cwd()
    if old != REPO_ROOT:
        os.chdir(REPO_ROOT)
    return old


def restore_cwd(old):
    """还原 ensure_repo_cwd() 之前的 cwd。"""
    if old is not None:
        os.chdir(old)


@contextmanager
def repo_cwd():
    """上下文管理器：在仓库根下执行一段代码，退出时自动还原 cwd。"""
    old = ensure_repo_cwd()
    try:
        yield REPO_ROOT
    finally:
        restore_cwd(old)


# ------------------------------------------------------------------ 路径解析
def abs_path(path):
    """把用户给的相对路径按**当前 cwd** 解析成绝对路径（切 cwd 之前调用）。"""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def resolve_checkpoint(name=DEFAULT_CKPT_NAME):
    """解析 --checkpoint 参数，返回真实存在的权重路径。

    支持：绝对路径、相对当前 cwd 的路径、demo/ 或 yoloe/ 下的相对路径、
    以及只给文件名（到 pretrain/ 里找）。找不到时给出可执行的修复命令。
    """
    p = Path(name).expanduser()
    candidates = []
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [p, DEMO_ROOT / p, REPO_ROOT / p, PRETRAIN_DIR / p.name, PRETRAIN_DIR / p]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    hint = (
        f"权重不存在: {name}\n"
        f"已查找: " + ", ".join(str(c) for c in candidates) + "\n"
        f"下载到 pretrain/ 的命令:\n"
        f'    hf download jameslahm/yoloe {p.name} --local-dir "{PRETRAIN_DIR}"'
    )
    raise FileNotFoundError(hint)


def resolve_tracker(spec):
    """解析 --tracker 参数，返回可以直接喂给 model.track() 的字符串。

    - 内置名（bytetrack.yaml / botsort.yaml）：原样返回，ultralytics 会在自己的
      cfg/trackers/ 下 glob 到它（见 ultralytics/utils/checks.py 的 check_file）。
    - 本地 yaml：必须转成**绝对路径**。check_file 只在「路径真实存在」时才按原样
      使用，否则一律去 ultralytics 包目录里 glob，本地文件会被判为找不到。
    """
    s = str(spec).strip()
    p = Path(s).expanduser()
    if p.is_absolute():
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"跟踪器配置不存在: {p}")

    if p.suffix.lower() in (".yaml", ".yml"):
        for cand in (p, DEMO_ROOT / p, REPO_ROOT / p):
            if cand.is_file():
                return str(cand.resolve())
        if p.name in BUILTIN_TRACKERS:
            return s
        raise FileNotFoundError(
            f"跟踪器配置不存在: {s}\n"
            f"已查找: {p}, {DEMO_ROOT / p}, {REPO_ROOT / p}\n"
            f"内置可用名: {', '.join(BUILTIN_TRACKERS)}"
        )
    return s


# ------------------------------------------------------------------ 打印辅助
def _display_width(text):
    """终端显示宽度：中文/全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, width, align="left"):
    """按显示宽度补齐，让中英文混排的表格能对齐。"""
    space = max(0, width - _display_width(text))
    return text + " " * space if align == "left" else " " * space + text


def kv(label, value, width=12):
    """对齐打印「标签 : 值」。"""
    print(f"{pad(label, width)}: {value}")


def section(title, width=60):
    """打印 = 分隔的标题块。"""
    print("=" * width)
    print(title)
    print("=" * width)


def hr(width=60, ch="-"):
    """打印一条分隔线。"""
    print(ch * width)
