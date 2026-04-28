## Project
LLM steering experiment (activation engineering) using HuggingFace transformers.

## Stack
Python, PyTorch, HuggingFace Transformers. Primary model: `meta-llama/Llama-3.1-8B-Instruct`.

## Setup
- HuggingFace token required at `~/.cache/huggingface/token` for gated model access
- `requirements.txt` uses conda-build URLs; conda environment recommended over standard pip

## Commands
- Compute steering vectors: `python steering-vectors/compute-steering-vectors-llama3.1.py`
- Steering vector outputs: `steering-vectors/meta-llama/Llama-3.1-8B-Instruct/{category}.pt`
- Demo notebooks: `notebooks/` (Jupyter)

## Architecture
- `src/models.py`: `_BaseModel` abstract class, `LLamaModel` wrapper
- `src/hooks.py`: `SteeringHook` injects activations into residual stream
- `src/transformers.py`: Model/tokenizer loading utilities
- `src/config.py`: Device detection, model constants, project paths

## Quirks
- MPS (Apple Silicon) memory: use `mps.synchronize()` + `mps.empty_cache()` after GPU operations
- No test suite, linting, or CI configured
- `datasets/`, `papers/`, and `PROJECT-STRUCTURE.md` are gitignored
