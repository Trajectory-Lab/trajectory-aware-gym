# Local Ollama Setup — Qwen3 Base Models

Guide for installing Qwen3-1.7B-Base and Qwen3-4B-Base locally via Ollama on WSL2 (Ubuntu) with NVIDIA GPU.

> **Note:** This has been tested on WSL2 + RTX 4090. macOS should work similarly but has not been tested yet.

---

## Prerequisites

- **WSL2** with Ubuntu (confirm with `wsl --list --verbose` in PowerShell)
- **NVIDIA GPU** visible in WSL (confirm with `nvidia-smi` inside WSL)
- **HuggingFace account** with an access token ([create one here](https://huggingface.co/settings/tokens))

---

## Step 1: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify GPU is detected in the output ("Nvidia GPU detected"). The API will be available at `http://127.0.0.1:11434`.

If Ollama was previously installed, this will update it to the latest version. After updating, restart the service:

```bash
sudo systemctl restart ollama
```

---

## Step 2: Install HuggingFace CLI

```bash
pip install huggingface-hub --break-system-packages
```

The CLI binary installs to `~/.local/bin/hf`. If not already in your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Then log in:

```bash
hf auth login
```

Paste your HuggingFace token when prompted.

---

## Step 3: Download GGUF Model Files

We use GGUF files from [mradermacher](https://huggingface.co/mradermacher), a well-known community quantizer who converts official Qwen safetensors to GGUF format using standard llama.cpp tools.

```bash
# Qwen3-1.7B-Base (~1.1GB)
hf download mradermacher/Qwen3-1.7B-Base-GGUF Qwen3-1.7B-Base.Q4_K_M.gguf --local-dir ~/models/gguf

# Qwen3-4B-Base (~2.5GB)
hf download mradermacher/Qwen3-4B-Base-GGUF Qwen3-4B-Base.Q4_K_M.gguf --local-dir ~/models/gguf
```

Source repos:
- [mradermacher/Qwen3-1.7B-Base-GGUF](https://huggingface.co/mradermacher/Qwen3-1.7B-Base-GGUF) (quantized from [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base))
- [mradermacher/Qwen3-4B-Base-GGUF](https://huggingface.co/mradermacher/Qwen3-4B-Base-GGUF) (quantized from [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base))

---

## Step 4: Register Models with Ollama

Create Modelfiles and register:

```bash
# Create Modelfiles
echo 'FROM /home/$USER/models/gguf/Qwen3-1.7B-Base.Q4_K_M.gguf' > ~/models/Modelfile-1.7b-base
echo 'FROM /home/$USER/models/gguf/Qwen3-4B-Base.Q4_K_M.gguf' > ~/models/Modelfile-4b-base

# Register with Ollama
ollama create qwen3-1.7b-base -f ~/models/Modelfile-1.7b-base
ollama create qwen3-4b-base -f ~/models/Modelfile-4b-base
```

You should see `success` for each. Verify:

```bash
ollama list
```

Both `qwen3-1.7b-base` and `qwen3-4b-base` should appear.

---

## Step 5: Test

**Interactive mode** (Ctrl+C to stop — base models don't know when to stop generating):

```bash
ollama run qwen3-1.7b-base "The capital of France is"
# Press Ctrl+C after you see enough output
```

**API mode** (recommended — lets you control output length):

```bash
# Qwen3-1.7B-Base
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-1.7b-base",
  "prompt": "The capital of France is",
  "stream": false,
  "options": {"num_predict": 50}
}'

# Qwen3-4B-Base
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-4b-base",
  "prompt": "The capital of France is",
  "stream": false,
  "options": {"num_predict": 50}
}'
```

---

## Using in Python

```python
import requests

def ollama_generate(model, prompt, max_tokens=256):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    })
    return response.json()["response"]

print(ollama_generate("qwen3-1.7b-base", "The capital of France is", max_tokens=50))
print(ollama_generate("qwen3-4b-base", "The capital of France is", max_tokens=50))
```

---

## Using with the Smoke Runner

The project's smoke runner uses LiteLLM with the `ollama/` prefix for base models:

```bash
# Single episode with Qwen3-1.7B-Base
uv run python scripts/run_gem_episode.py \
  --smoke \
  --task-model-id "ollama/qwen3-1.7b-base" \
  --environment "math:Orz57K" \
  --max-steps 1 \
  --max-response-tokens 4096 \
  --temperature 0.0 \
  --seed 42 \
  --show-log

# Or using the quick-test experiment config (already configured for base model)
uv run python scripts/run_gem_episode.py \
  --smoke \
  --experiment-config experiments/quick-test/config.yaml \
  --max-steps 1 \
  --show-log
```

---

## Notes

- **Base models are completion models**, not chat models. They continue text from your prompt rather than answering questions. This is expected behavior for GEPA/GRPO experiments.
- **Q4_K_M quantization** is a good balance of speed and quality. The 4090 can easily handle both models. Quantization introduces minor differences vs. the full-precision weights running on AWS SageMaker, but this does not affect relative comparisons between methods (GEPA vs GRPO) since both use the same quantized model.
- **Ollama runs a persistent server** at `localhost:11434`. It starts automatically via systemd. If it's not running: `sudo systemctl start ollama`.
- **GPU memory usage**: 1.7B ≈ 5GB VRAM, 4B ≈ 7GB VRAM. Both fit comfortably on a 24GB RTX 4090, even simultaneously.
- **Stop sequences**: The episode runner automatically adds stop sequences (`<|endoftext|>`, `<|im_end|>`, etc.) for `ollama/` base models to prevent infinite text generation.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ollama: command not found` | Re-run the install script |
| `unsupported architecture` | Update Ollama: `curl -fsSL https://ollama.com/install.sh \| sh` then `sudo systemctl restart ollama` |
| `hf: command not found` | Add to PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc` |
| `401 Unauthorized` on HF download | Run `hf auth login` and paste a valid token |
| Ollama server not responding | `sudo pkill -9 ollama && sleep 2 && ollama serve &` |
| Model generates forever | Use API with `num_predict` option, or Ctrl+C in interactive mode |
