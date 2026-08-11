# CPAT

Credit portfolio analysis tool.

## Documentation

Docs are generated from docstrings (NumPy style) with [pdoc](https://pdoc.dev).

```bash
pip install -e ".[docs]"

# live preview while writing, auto-reloads on save
pdoc src.inputs src.models

# build a static HTML site into ./site (gitignored)
pdoc src.inputs src.models -o site
```