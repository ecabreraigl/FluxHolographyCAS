# Quick Start Guide

**New to FluxHolographyCAS?** This 2-minute guide gets you running.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/ecabreraigl/FluxHolographyCAS.git
cd FluxHolographyCAS
```

---

## Step 2: Install Dependencies

### Recommended: Simple Installation (works for everyone)

```bash
pip install -r requirements.txt
```

That's it! This installs:
- `sympy` (symbolic math)
- `numpy` (numerical computing)

### Alternative: Full Package Installation

If you have pip ≥ 21.3 and want the package installed:

```bash
pip install -e .
```

If you get an error, **use the simple method above instead**.

### Alternative: Using Conda

If you prefer conda:

```bash
conda env create -f environment.yml
conda activate flux-holography
```

---

## Step 3: Verify Installation

```bash
python cas/fh_master_cas.py
```

You should see:

```
Flux Holography — Master CAS Backbone Summary

Global all_pass: True

[Sectors]
---------
core: all_pass=True, count=5, failed=[]
...
```

**Success!** ✓ You're ready to use FH CAS.

---

## Step 4: Try the Examples

### Test a New Prediction

```bash
cd examples/predictions
python test_new_prediction.py
```

### Compare with Observations

```bash
cd examples/observations
python compare_with_observations.py
```

### Run the Jupyter Notebook

```bash
jupyter notebook FH_backbone.ipynb
```

Or use [Google Colab](https://colab.research.google.com/github/ecabreraigl/FluxHolographyCAS/blob/main/FH_backbone.ipynb) (no installation needed!)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sympy'"

**Fix:**
```bash
pip install -r requirements.txt
```

### "Permission denied" errors

**Fix:**
```bash
# Install in user directory
pip install --user -r requirements.txt
```

### Still having issues?

1. Check you have Python ≥ 3.8: `python --version`
2. See detailed [INSTALL.md](INSTALL.md)
3. Open an [issue](https://github.com/ecabreraigl/FluxHolographyCAS/issues)

---

## What Next?

- 📖 Read [examples/predictions/README.md](examples/predictions/README.md) to test new predictions
- 📊 Read [examples/observations/README.md](examples/observations/README.md) to compare with data
- 🔬 Explore the CAS modules in `cas/`
- 📚 Read the FH papers (see `cas/fh_master_cas.py` for citations)

---

## Quick Reference

```bash
# Installation (choose one)
pip install -r requirements.txt              # Recommended
pip install -e .                             # Full package (if pip >= 21.3)
conda env create -f environment.yml          # Conda users

# Run CAS verification
python cas/fh_master_cas.py

# Run examples
python examples/predictions/test_new_prediction.py
python examples/observations/compare_with_observations.py

# Interactive notebook
jupyter notebook FH_backbone.ipynb
```

---

**Total time:** ~2 minutes | **Difficulty:** Easy

Happy exploring! 🚀
