# tiny-llm

PyTorch でゼロから実装する GPT 系 LLM。

```bash
uv sync                    # 本体 + dev
uv sync --group bonus      # ボーナス教材用の依存も入れる
uv run python check_gpu.py # GPU 確認
uv run pytest              # 環境テスト
uv run jupyter lab
```
