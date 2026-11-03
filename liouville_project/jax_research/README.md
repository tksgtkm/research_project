# JAX (CUDA 12) 環境構築メモ

GTX 1080 ×2 の Ubuntu 24.04 マシンで、uv を使って JAX の GPU 環境を作る手順。

## 前提となるマシン構成

| 項目 | 内容 |
| --- | --- |
| OS | Ubuntu 24.04 (HWE カーネル 7.0.0-34-generic) |
| CPU | Intel Core i9-7900X (10C/20T, AVX-512) |
| メモリ | 16 GB |
| GPU | GeForce GTX 1080 ×2 (Pascal, SM 6.1, 8 GB, PCIe Gen3 x16) |
| ドライバ | nvidia-driver-580 (580.178.04, Ubuntu 公式版 + ビルド済みモジュール) |

### ドライバまわりの前提

- ドライバは Ubuntu 公式版(`nvidia-driver-580` + `linux-modules-nvidia-580-generic-hwe-24.04`)を使う。
- CUDA リポジトリ版のドライバが混ざらないよう、以下の pin を残しておくこと。

```text
# /etc/apt/preferences.d/nvidia-driver-from-ubuntu
Package: nvidia-* libnvidia-* xserver-xorg-video-nvidia-* libxnvctrl*
Pin: origin developer.download.nvidia.com
Pin-Priority: 100
```

確認コマンド:

```bash
nvidia-smi   # Driver Version: 580.x で GPU 2 枚が見えること
```

## 1. uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # 現在のシェルに PATH を反映
uv --version
```

- `~/.local/bin` に入るので sudo は不要
- 更新は `uv self update`

## 2. プロジェクトの作成

```bash
mkdir -p ~/work/jax-lab && cd ~/work/jax-lab
uv init
uv python pin 3.12
```

## 3. JAX (CUDA 12 版) のインストール

```bash
uv add "jax[cuda12]"
uv run python -c "import jax; print(jax.__version__); print(jax.devices())"
```

期待される出力: `[CudaDevice(id=0), CudaDevice(id=1)]`

> **重要:** `jax[cuda13]` は使わない。CUDA 13 は SM 7.5 以上のみ対応で、GTX 1080 (SM 6.1) は対象外。
> CUDA 12 版の JAX は SM 5.2 以上に対応している。

- CUDA / cuDNN は pip の wheel として同梱されるため、システムの CUDA Toolkit は不要
- `LD_LIBRARY_PATH` にシステムの CUDA (例: `/usr/lib/cuda`) を通していると古いライブラリを掴むことがあるので、通さない
- バージョンは `pyproject.toml` / `uv.lock` に記録される。`uv.lock` はコミットしておく

## 4. JupyterLab 用カーネルの登録

```bash
uv add --dev ipykernel
uv run python -m ipykernel install --user --name jax-cuda12 --display-name "JAX (CUDA12)"
```

ノートブック先頭(`import jax` より前)で環境変数を設定する:

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # GPU メモリの事前確保を無効化
os.environ["CUDA_VISIBLE_DEVICES"] = "0"               # 画面表示に使っていない GPU 0 を使う

import jax
jax.devices()
```

スクリプト実行時:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python script.py
```

## 5. 運用上の注意

- **精度:** Pascal はテンソルコアがなく FP16 / bf16 が遅いので、float32 を基本にする
- **ホストメモリ:** 16 GB と少ない。データは一括でロードせず、ストリーミングで読み込む設計にする
- **GPU の役割分担:** GPU 1 (`65:00.0`) は画面表示を兼ねているため、計算は GPU 0 に寄せると安定する
- **再現性:** `uv.lock` で依存を固定しておけば、Colab や新しいマシンへの移行が容易
- **サポート期限:**
  - 580 系ドライバは Pascal 向けの最終ブランチ (2028 年頃までの見込み)
  - JAX の CUDA 12 版配布がいつまで続くかは未定。動く組み合わせはバージョンを固定しておく

## 6. トラブル時の確認コマンド

```bash
nvidia-smi                                    # ドライバと GPU の状態
dkms status                                   # DKMS モジュールの状態
uname -r                                      # 起動中のカーネル
uv run python -c "import jax; print(jax.devices())"
```