# Dependency Management Guide

This document explains the dependency management approach for FluxHolographyCAS.

---

## Overview

FluxHolographyCAS uses a **multi-tier dependency management** system to support different workflows:

| File | Purpose | Use Case |
|------|---------|----------|
| `pyproject.toml` | Modern standard (PEP 517/518) | Primary configuration |
| `requirements.txt` | Simple dependency list | Quick installation |
| `requirements-dev.txt` | Development tools | Contributors |
| `environment.yml` | Conda environment | Scientific computing workflows |
| `setup.py` | Backward compatibility | Older pip versions |

---

## Recommended Installation Methods

### For End Users (Quickest)

```bash
# Just install dependencies
pip install -r requirements.txt
```

This installs only what you need to run the CAS:
- `sympy>=1.12` - Symbolic mathematics
- `numpy>=1.21.0` - Numerical computing

### For Package Installation (Most Features)

```bash
# Install as a package (requires pip >= 21.3)
pip install -e .
```

Benefits:
- Makes `cas` module importable from anywhere
- Installs `fh-cas` command-line tool
- Enables optional dependencies: `pip install -e ".[viz]"`

**Note**: If you have an older pip version, you'll see an error. Either:
1. Upgrade pip: `pip install --upgrade pip`
2. Use requirements.txt method instead

### For Developers

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

This includes:
- Testing: `pytest`, `pytest-cov`
- Code quality: `black`, `flake8`, `mypy`
- Interactive: `jupyter`, `ipython`
- Visualization: `matplotlib`, `seaborn`

### For Conda Users

```bash
conda env create -f environment.yml
conda activate flux-holography
```

Creates a complete scientific Python environment.

---

## Dependency Specifications

### Core Dependencies (Minimal)

These are **required** to run any FH CAS code:

```toml
sympy>=1.12     # Symbolic computation engine
numpy>=1.21.0   # Numerical arrays and functions
```

**Why these versions?**
- `sympy>=1.12`: Modern API, better simplification algorithms
- `numpy>=1.21.0`: Type hints support, better compatibility

### Optional Dependencies

Install via `pip install -e ".[group]"`:

#### Visualization (`viz`)
```toml
matplotlib>=3.5  # Plotting
seaborn>=0.12    # Statistical visualizations
```

Use for: Creating plots in `examples/observations/`

#### Development (`dev`)
```toml
pytest>=7.0       # Testing framework
pytest-cov>=4.0   # Coverage reports
black>=23.0       # Code formatter
flake8>=6.0       # Style checker
mypy>=1.0         # Type checker
ipython>=8.0      # Interactive shell
jupyter>=1.0      # Notebooks
```

Use for: Contributing code, running tests, development

#### Documentation (`docs`)
```toml
sphinx>=5.0           # Documentation generator
sphinx-rtd-theme>=1.0 # ReadTheDocs theme
myst-parser>=0.18     # Markdown in docs
```

Use for: Building documentation (future)

#### All (`all`)
Installs everything above.

---

## Files Explained

### `pyproject.toml` (Primary)

**Purpose**: Modern Python package configuration (PEP 517/518)

**Contains**:
- Project metadata (name, version, authors)
- Dependencies (core and optional)
- Tool configuration (black, pytest, mypy)
- Build system requirements

**Advantages**:
- Single source of truth
- Standard across Python ecosystem
- Supports optional dependencies
- Includes tool configurations

**Example**:
```toml
[project]
name = "flux-holography-cas"
version = "0.1.0"
dependencies = ["sympy>=1.12", "numpy>=1.21.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black>=23.0"]
```

### `requirements.txt` (Backward Compatibility)

**Purpose**: Simple pip dependency list

**Contains**:
```
sympy>=1.12
numpy>=1.21.0
```

**When to use**:
- You just want to run the CAS
- You don't need package installation
- You have an old pip version
- You're in a restricted environment

**Advantages**:
- Simple and universal
- Works with any pip version
- Easy to understand
- Quick installation

### `requirements-dev.txt` (Development)

**Purpose**: Development and testing tools

**Contains**:
- Core dependencies (via `-r requirements.txt`)
- Testing tools
- Code quality tools
- Interactive tools
- Visualization libraries

**When to use**:
- Contributing to the project
- Running tests
- Developing new features
- Debugging

**Install**:
```bash
pip install -r requirements-dev.txt
```

### `environment.yml` (Conda)

**Purpose**: Conda environment specification

**Contains**:
- Python version
- All dependencies (conda and pip)
- Channel specifications

**When to use**:
- You use Anaconda/Miniconda
- You want complete environment reproducibility
- You're doing scientific computing
- You need non-Python dependencies

**Advantages**:
- Reproducible environments
- Includes Python version
- Better for scientific packages
- Cross-platform

**Create environment**:
```bash
conda env create -f environment.yml
conda activate flux-holography
```

### `setup.py` (Legacy Compatibility)

**Purpose**: Backward compatibility with older tools

**Contains**:
```python
from setuptools import setup
setup()  # Configuration in pyproject.toml
```

**When needed**:
- pip < 21.3
- Legacy build tools
- Certain CI systems

**Note**: This is a minimal shim. All actual configuration is in `pyproject.toml`.

---

## Version Pinning Philosophy

### Core Dependencies: Minimum Versions

```
sympy>=1.12
numpy>=1.21.0
```

**Why `>=` (not `==`)?**
- Allows users to get bug fixes
- Prevents dependency conflicts
- Trusts upstream semantic versioning

**When to pin exactly?**
- Only in production deployments
- Not in library code like FH CAS

### Development Dependencies: Flexible

```
pytest>=7.0
black>=23.0
```

**Why flexible?**
- Development tools update frequently
- Users may have existing installations
- Newer versions usually backward compatible

### Lock Files (Not Included)

We don't include `poetry.lock` or `Pipfile.lock` because:
- This is a **library**, not an application
- Users should get latest compatible versions
- Lock files would conflict with user environments

**Exception**: For reproducible research, you can create:
```bash
pip freeze > requirements-lock.txt
```

---

## Troubleshooting

### Issue: "File setup.py not found"

**Cause**: Old pip version with pyproject.toml

**Solution**:
```bash
# Option 1: Upgrade pip
pip install --upgrade pip
pip install -e .

# Option 2: Use requirements.txt
pip install -r requirements.txt
```

### Issue: "Could not find a version that satisfies"

**Cause**: Python version too old

**Solution**:
```bash
# Check Python version
python --version

# Need Python >= 3.8
# Upgrade or use pyenv/conda
```

### Issue: Permission denied

**Cause**: Trying to install system-wide

**Solution**:
```bash
# Option 1: User install
pip install --user -e .

# Option 2: Virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue: Dependency conflict

**Cause**: Existing packages conflict with FH CAS

**Solution**:
```bash
# Use a fresh virtual environment
python -m venv fh-env
source fh-env/bin/activate
pip install -e .

# Or conda
conda create -n flux-holography python=3.10
conda activate flux-holography
pip install -e .
```

---

## Verifying Installation

After installation, verify dependencies:

```bash
# Check installed packages
pip list | grep -E "sympy|numpy"

# Should show:
# numpy        2.0.2
# sympy        1.14.0

# Test imports
python -c "import sympy, numpy; print('Dependencies OK')"

# Run CAS verification
python cas/fh_master_cas.py
```

Expected output:
```
Flux Holography — Master CAS Backbone Summary

Global all_pass: True
...
```

---

## Updating Dependencies

### Update to Latest Compatible Versions

```bash
pip install --upgrade -r requirements.txt
```

### Update Development Tools

```bash
pip install --upgrade -r requirements-dev.txt
```

### Update Everything in Conda

```bash
conda update --all
```

### Check for Outdated Packages

```bash
pip list --outdated
```

---

## Adding New Dependencies

### For Core Functionality

1. Add to `pyproject.toml`:
```toml
dependencies = [
    "sympy>=1.12",
    "numpy>=1.21.0",
    "your-new-package>=X.Y.Z",  # Add here
]
```

2. Add to `requirements.txt`:
```
sympy>=1.12
numpy>=1.21.0
your-new-package>=X.Y.Z  # Add here
```

3. Add to `environment.yml`:
```yaml
dependencies:
  - your-new-package>=X.Y.Z  # Add here
```

### For Optional Features

Add to `pyproject.toml` optional dependencies:

```toml
[project.optional-dependencies]
special-feature = [
    "special-package>=1.0",
]
```

Users can then install:
```bash
pip install -e ".[special-feature]"
```

---

## Best Practices

### ✓ DO

1. **Use virtual environments** - Isolate dependencies
2. **Pin minimum versions** - Use `>=` for flexibility
3. **Test across Python versions** - 3.8, 3.9, 3.10, 3.11
4. **Document new dependencies** - Explain why they're needed
5. **Keep dependencies minimal** - Only add what's necessary

### ✗ DON'T

1. **Don't pin exact versions** (in libraries)
2. **Don't add unnecessary dependencies**
3. **Don't forget to update all files** when adding deps
4. **Don't install system-wide** without virtual env
5. **Don't ignore dependency conflicts**

---

## For Contributors

When adding a new dependency:

1. **Ask**: Is it really necessary?
2. **Research**: Is it maintained? License compatible?
3. **Document**: Update this file and INSTALL.md
4. **Test**: Ensure it works on all platforms
5. **Communicate**: Open an issue to discuss first

Example pull request:
```markdown
## Adding scipy for numerical integration

### Motivation
Need `scipy.integrate.quad` for numerical integrals in FH-VI

### Changes
- Added `scipy>=1.9` to core dependencies
- Updated pyproject.toml, requirements.txt, environment.yml
- Added example in examples/predictions/numerical_integration.py

### Testing
- Tested on Python 3.8, 3.9, 3.10, 3.11
- Works on Linux, macOS, Windows
- No conflicts with existing dependencies
```

---

## Future Improvements

Potential enhancements to dependency management:

1. **CI/CD Matrix Testing**: Test against multiple Python versions automatically
2. **Dependabot**: Automated dependency updates
3. **Lock Files for Testing**: Reproducible test environments
4. **Docker Images**: Pre-built environments
5. **PyPI Publishing**: `pip install flux-holography-cas`

---

## Questions?

For dependency-related issues:

1. Check this guide
2. See [INSTALL.md](INSTALL.md) for installation help
3. Open an issue with your `pip list` output

---

**Last Updated**: 2024-11-30
