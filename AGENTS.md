## Project
LLM steering experiment (activation engineering) using HuggingFace transformers.

## Stack
Python, PyTorch, HuggingFace Transformers. Primary model: `meta-llama/Llama-3.1-8B-Instruct`.

## Setup
- HuggingFace token required at `~/.cache/huggingface/token` for gated model access
- `requirements.txt` uses conda-build URLs; conda environment recommended over standard pip

## Commands
- Compute steering vectors: `python steering-vectors/compute-steering-vectors-llama3.1.py` (add `--force-computing` to recompute existing vectors)
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
- Steering vector computation uses last token hidden state (hidden_states[-1])
- Hook cleanup: clear `module._forward_pre_hooks`/`_forward_hooks` dicts, or properly track `RemovableHandle` objects (do not call `remove()` on hook functions)
- Layer targeting for hooks: use `str(LAYER_TO_STEER)` (e.g., "16") when matching layer names in `model.layers`
- `test-using-steering.ipynb` Cell 7 has known hook registration bugs; revised version uses proper handle management
