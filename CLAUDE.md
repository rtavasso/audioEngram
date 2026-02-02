# Project Configuration

## Running Python Files

Always run Python files using uv:

```bash
uv run python <script.py>
```

This ensures the correct virtual environment (.venv) and dependencies are used.

## Dependencies

Main dependencies are managed through pyproject.toml. PyTorch with CUDA 12.4 support is installed automatically via uv.

### Moshi Submodule

Moshi (the Mimi VAE) is included as a git submodule. After cloning, initialize it:

```bash
git submodule update --init --recursive
```

Then install in editable mode:

```bash
uv pip install -e ./moshi/moshi --python .venv/bin/python
```

## GPU Support

The project is configured to install PyTorch with CUDA 12.4 support. Verify with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
