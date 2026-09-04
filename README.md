# CPAT

Credit portfolio analysis tool.

## List of tools to build

- [x] Binary default monte carlo loss distribution
- [ ] Mark-to-market loan pricer
- [ ] Mark-to-market CDS pricer
- [x] CDO tranche loss distributions
- [ ] Multi period analytical EL
- [ ] Macro correlation matrix
- [ ] Conditional loss distribution simulator
- [ ] Multi period stress testing tool
- [x] PD-LGD correlation
- [x] Student-t copula tail dependence
- [ ] Risk free rate bootstrappers

## Documentation

Docs are generated from docstrings (NumPy style) with [pdoc](https://pdoc.dev).

```bash
pip install -e ".[docs]"

# live preview while writing, auto-reloads on save
pdoc src.inputs src.results src.models

# build a static HTML site into ./site (gitignored)
pdoc src.inputs src.results src.models -o site
```