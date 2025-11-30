# Installation Guide

This guide covers multiple ways to install and use the Flux Holography CAS.

---

## Quick Start

### Option 1: Pip Install (Recommended)

For most users, the simplest approach:

```bash
# Clone the repository
git clone https://github.com/ecabreraigl/FluxHolographyCAS.git
cd FluxHolographyCAS

# Install in editable mode with pip
pip install -e .
```

This makes the `cas` module importable from anywhere and installs the `fh-cas` command-line tool.

### Option 2: Requirements.txt

If you just want to run the examples without installing the package:

```bash
# Clone the repository
git clone https://github.com/ecabreraigl/FluxHolographyCAS.git
cd FluxHolographyCAS

# Install dependencies only
pip install -r requirements.txt
```

### Option 3: Conda Environment (For Scientific Computing)

If you use conda (common in scientific computing):

```bash
# Clone the repository
git clone https://github.com/ecabreraigl/FluxHolographyCAS.git
cd FluxHolographyCAS

# Create and activate conda environment
conda env create -f environment.yml
conda activate flux-holography
```

---

## Detailed Installation Options

### For End Users

**Use Case**: You want to use FH CAS for your research.

```bash
# Install the package
pip install -e .

# Verify installation
python -c "import cas.fh_master_cas as fh; print('FH CAS installed successfully!')"

# Run the master verification
fh-cas
```

Or with specific optional dependencies:

```bash
# With visualization support
pip install -e ".[viz]"

# With documentation tools
pip install -e ".[docs]"

# With everything
pip install -e ".[all]"
```

### For Developers

**Use Case**: You want to contribute to FH CAS development.

```bash
# Clone the repo
git clone https://github.com/ecabreraigl/FluxHolographyCAS.git
cd FluxHolographyCAS

# Install with development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e ".[dev]"

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

**Development workflow:**

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=cas --cov-report=html

# Format code
black cas/ examples/

# Check code style
flake8 cas/

# Type checking
mypy cas/
```

### For Jupyter Notebook Users

**Use Case**: You want to explore FH interactively.

```bash
# Install with Jupyter support
pip install -e ".[dev]"

# Or via conda
conda env create -f environment.yml
conda activate flux-holography

# Launch Jupyter
jupyter notebook FH_backbone.ipynb
```

Or use Google Colab (no installation needed):
- Click the "Open in Colab" badge in the README

---

## Verification

After installation, verify everything works:

```bash
# Test 1: Import CAS modules
python -c "from cas import fh_core_cas, fh_master_cas; print('✓ Imports work')"

# Test 2: Run master CAS
python cas/fh_master_cas.py

# Test 3: Run examples
python examples/predictions/test_new_prediction.py
python examples/observations/compare_with_observations.py
```

Expected output should show:
```
✓ Imports work
Flux Holography — Master CAS Backbone Summary

Global all_pass: True
...
```

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 512 MB (symbolic computation can be memory-intensive)
- **Disk**: ~50 MB for repository and dependencies

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 2 GB+
- **Disk**: 1 GB+ (for development tools and docs)

### Platform Support
- ✓ **Linux**: Tested on Ubuntu 20.04+, Debian, Fedora
- ✓ **macOS**: Tested on macOS 11+ (both Intel and Apple Silicon)
- ✓ **Windows**: Should work, but less tested (use WSL2 for best results)

---

## Dependency Details

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `sympy` | ≥1.12 | Symbolic mathematics engine |
| `numpy` | ≥1.21 | Numerical computations |

These are minimal - the CAS is designed to be lightweight.

### Optional Dependencies

**Visualization** (`pip install -e ".[viz]"`):
- `matplotlib` - Plotting predictions vs observations
- `seaborn` - Statistical visualizations

**Development** (`pip install -e ".[dev]"`):
- `pytest` - Testing framework
- `pytest-cov` - Code coverage
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Static type checker
- `jupyter` - Interactive notebooks

**Documentation** (`pip install -e ".[docs]"`):
- `sphinx` - Documentation generator
- `sphinx-rtd-theme` - ReadTheDocs theme
- `myst-parser` - Markdown support in docs

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sympy'`

**Solution:**
```bash
pip install sympy numpy
```

### Issue: `ImportError: cannot import name 'fh_master_cas'`

**Solution:**
```bash
# Make sure you're in the repository root
cd FluxHolographyCAS

# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Permission errors when installing

**Solution:**
```bash
# Install in user directory (no sudo needed)
pip install --user -e .

# Or use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Issue: Conda environment conflicts

**Solution:**
```bash
# Remove and recreate environment
conda env remove -n flux-holography
conda env create -f environment.yml
conda activate flux-holography
```

### Issue: Old package version installed

**Solution:**
```bash
# Uninstall old version
pip uninstall flux-holography-cas

# Reinstall
pip install -e .

# Or update
pip install --upgrade -e .
```

---

## Virtual Environment Best Practices

### Using venv (Built-in)

```bash
# Create virtual environment
python -m venv fh-env

# Activate (Linux/macOS)
source fh-env/bin/activate

# Activate (Windows)
fh-env\Scripts\activate

# Install dependencies
pip install -e .

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create from environment.yml
conda env create -f environment.yml

# Or create manually
conda create -n flux-holography python=3.10
conda activate flux-holography
pip install -e .

# List environments
conda env list

# Remove environment
conda env remove -n flux-holography
```

---

## Installing from PyPI (Future)

Once published to PyPI, installation will be even simpler:

```bash
# Not yet available - coming soon
pip install flux-holography-cas
```

---

## Docker (Advanced)

For reproducible environments across platforms:

```dockerfile
# Dockerfile (example)
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[all]"

CMD ["fh-cas"]
```

```bash
# Build and run
docker build -t flux-holography .
docker run -it flux-holography
```

---

## Offline Installation

For systems without internet access:

```bash
# On connected machine: download packages
pip download -d packages -r requirements.txt

# Transfer 'packages/' directory to offline machine

# On offline machine: install from local directory
pip install --no-index --find-links=packages -r requirements.txt
```

---

## Updating

### Update to latest version

```bash
# Pull latest changes
git pull origin main

# Reinstall (picks up new dependencies if any)
pip install -e .
```

### Update dependencies only

```bash
pip install --upgrade -r requirements.txt
```

---

## Uninstallation

```bash
# Uninstall package
pip uninstall flux-holography-cas

# Remove conda environment
conda env remove -n flux-holography

# Remove virtual environment
rm -rf venv/  # or fh-env/
```

---

## Getting Help

If you encounter installation issues:

1. Check this troubleshooting guide first
2. Search [existing issues](https://github.com/ecabreraigl/FluxHolographyCAS/issues)
3. Open a new issue with:
   - Your OS and Python version
   - The full error message
   - Output of `pip list` or `conda list`

---

## Next Steps

After successful installation:

1. ✓ Run the master CAS: `python cas/fh_master_cas.py`
2. ✓ Try examples: `cd examples/predictions && python test_new_prediction.py`
3. ✓ Read the methodology: Check `examples/*/README.md`
4. ✓ Explore notebooks: `jupyter notebook FH_backbone.ipynb`
5. ✓ Start developing: See `CONTRIBUTING.md` (if available)

**Happy computing!** 🚀
