#!/usr/bin/env bash
# LLMs-from-scratch 相当の環境を、任意のプロジェクト名でゼロから構築する
#
# 使い方:
#   bash bootstrap.sh <project-name> [target-dir]
#   例) bash bootstrap.sh tiny-gpt            -> ./tiny-gpt/ に作成
#       bash bootstrap.sh tiny-gpt ~/work/llm -> ~/work/llm/ に作成
#
# 前提: uv がインストール済み (curl -LsSf https://astral.sh/uv/install.sh | sh)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash bootstrap.sh <project-name> [target-dir]" >&2
  exit 1
fi

DIST="$(echo "$1" | tr '[:upper:]_' '[:lower:]-')"   # 配布名 (ハイフン区切り)
MOD="${DIST//-/_}"                                   # import 名 (アンダースコア区切り)
DIR="${2:-$DIST}"
PY_VERSION="3.12"   # TensorFlow が <3.13 のため 3.12 に固定

if ! [[ "$DIST" =~ ^[a-z][a-z0-9-]*$ ]]; then
  echo "error: 名前は英小文字で始まり、英小文字・数字・ハイフンのみにしてください: $DIST" >&2
  exit 1
fi
command -v uv >/dev/null || { echo "error: uv が見つかりません" >&2; exit 1; }

mkdir -p "$DIR/pkg/$MOD" "$DIR/tests"
cd "$DIR"

echo "$PY_VERSION" > .python-version

cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "${DIST}"
version = "0.1.0"
description = "GPT-like LLM implemented from scratch in PyTorch"
readme = "README.md"
requires-python = ">=3.10,<3.15"
dependencies = [
  # Intel Mac は古い PyTorch wheel のみ
  "torch>=2.2.2,<2.6; sys_platform == 'darwin' and platform_machine == 'x86_64'",
  "torch>=2.6.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
  # Linux: GTX 1080 (Pascal, sm_61) を使うため cu126 ビルドに固定。
  # cu128/cu130 ビルドは sm_75 以上のみ。cu126 wheel は 2.14 で最後。
  "torch>=2.6.0,<2.15; sys_platform == 'linux'",
  "torch>=2.6.0; sys_platform == 'win32'",
  # TensorFlow は GPT-2 の公式重み (TF checkpoint) の読み込み用。CPU 版で十分
  "tensorflow>=2.16.2; sys_platform == 'darwin' and platform_machine == 'x86_64' and python_version < '3.13'",
  "tensorflow>=2.18.0; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version < '3.13'",
  "tensorflow-cpu>=2.18.0; sys_platform == 'linux' and platform_machine == 'x86_64' and python_version < '3.13'",
  "tensorflow>=2.18.0; sys_platform == 'linux' and platform_machine == 'aarch64' and python_version < '3.13'",
  "tensorflow-cpu>=2.18.0; sys_platform == 'win32' and python_version < '3.13'",
  "jupyterlab>=4.0",
  "tiktoken>=0.5.1",
  "matplotlib>=3.7.1",
  "tqdm>=4.66.1",
  "numpy>=1.26",
  "pandas>=2.2.1",
  "pip>=25.0.1",
  "pytest>=9.1.1",
]

[tool.uv.sources]
${DIST} = { workspace = true }
torch = [
  { index = "pytorch-cu126", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true   # torch 以外はこのインデックスを見ない

[dependency-groups]
dev = [
    "build>=1.2.2.post1",
    "twine>=6.1.0",
    "tokenizers>=0.22.0",
    "safetensors>=0.6.2",
]
bonus = [
    "blobfile>=3.0.0",
    "chainlit>=1.2.0",
    "huggingface_hub>=0.34.4",
    "ipywidgets>=8.1.2",
    "${DIST}",
    "openai>=1.30.3",
    "requests",
    "safetensors>=0.6.2",
    "scikit-learn>=1.3.1",
    "sentencepiece>=0.1.99",
    "thop",
    "tokenizers>=0.21.1",
    "transformers>=5.0.0",
    "tqdm>=4.65.0",
]

[tool.ruff]
line-length = 140

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
exclude = [".venv"]
ignore = [
    "C406", "E226", "E402", "E702", "E703",
    "E722", "E731", "E741"
]

[tool.setuptools]
package-dir = {"" = "pkg"}

[tool.setuptools.packages.find]
where = ["pkg"]
EOF

cat > "pkg/$MOD/__init__.py" <<EOF
"""${DIST}: GPT-like LLM implemented from scratch."""

__version__ = "0.1.0"
EOF

cat > tests/test_env.py <<EOF
import importlib


def test_package_importable():
    mod = importlib.import_module("${MOD}")
    assert mod.__version__


def test_core_deps():
    import tiktoken
    import torch

    enc = tiktoken.get_encoding("gpt2")
    assert enc.decode(enc.encode("Hello, world")) == "Hello, world"
    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
EOF

cat > check_gpu.py <<'EOF'
"""GPU が PyTorch から使えるか確認する。"""
import torch

print("torch       :", torch.__version__)
print("CUDA build  :", torch.version.cuda)
print("available   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("arch list   :", torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"GPU{i}        : {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})")
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul OK   :", (x @ x).shape)
EOF

cat > README.md <<EOF
# ${DIST}

PyTorch でゼロから実装する GPT 系 LLM。

\`\`\`bash
uv sync                    # 本体 + dev
uv sync --group bonus      # ボーナス教材用の依存も入れる
uv run python check_gpu.py # GPU 確認
uv run pytest              # 環境テスト
uv run jupyter lab
\`\`\`
EOF

cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.egg-info/
.ipynb_checkpoints/
*.pth
*.safetensors
gpt2/
EOF

[[ -d .git ]] || git init -q

echo ">> uv sync (Python ${PY_VERSION})"
uv sync --python "$PY_VERSION"

echo
echo "完了: $(pwd)"
echo "  uv run python check_gpu.py   # GPU 認識の確認"
echo "  uv run pytest                # 環境テスト"