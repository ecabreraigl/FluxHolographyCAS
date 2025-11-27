# FluxHolographyCAS  
### A Computational Backbone for Flux Holography (FH)

This repository contains the full **CAS (Computer Algebra System) verification suite** for  
**Flux Holography**, together with ready-to-run Jupyter/Colab notebooks.

The CAS modules encode the *exact algebraic structure* of FH:
- Entropy–Action Law (EAL)  
- Universal Tick Law (UTL)  
- Universal Area Law (UAL)  
- Spacetime response constant \( k_{\mathrm{SEG}} = 4\pi G / c^3 \)  
- Horizon identities (Schwarzschild, de Sitter, FRW)  
- Tick sector + Planckian relaxation bound  
- Integrability / Iyer–Wald selection  
- FH corollaries (entropic inertia, Hubble quantum, dark-energy ratio, etc.)

Every identity is checked symbolically via SymPy.

---

# 🚀 Run the FH Backbone Notebook

Click to open directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecabreraigl/FluxHolographyCAS/blob/main/FH_backbone.ipynb)

This notebook performs:

- cloning this repository  
- importing the CAS suite  
- running the master consistency check  
- printing a sector-by-sector summary:

| Sector | Content |
|-------|---------|
| **Core Backbone** | EAL, flux law, UTL, UAL, \(k_{\mathrm{SEG}}\), \( \Theta \) |
| **Horizons & Cosmology** | Schwarzschild, de Sitter, FRW checks |
| **Tick Sector** | Universal Tick Law, Planckian bound |
| **Selection / Integrability** | Iyer–Wald invariance, rank-1 closure |
| **Corollaries** | Bekenstein shift, inertia, Hubble quantum, dark energy |

If the notebook reports  
**“All identities passed? True”**,  
the entire FH backbone is internally consistent.

---

# 📂 Repository Structure

FluxHolographyCAS/
│
├── FH_backbone.ipynb        # Main computational check (run this first)
│
├── cas/                     # CAS suite (Python symbolic modules)
│   ├── fh_core_cas.py
│   ├── fh_horizons_cosmo_cas.py
│   ├── fh_tick_noneq_cas.py
│   ├── fh_selection_integrability_iw_cas.py
│   ├── fh_corollaries_cas.py
│   ├── fh_complementary_cas.py
│   ├── fh_master_cas.py     # Orchestrates all checks
│   └── init.py
│
└── README.md

Each CAS module corresponds to a conceptual layer of FH.

---

# 🧠 What This Repository Provides

### ✔ A **transparent, constants-explicit** implementation of the FH backbone  
No hidden normalization choices, no missing factors of \(2\pi\), no geometric ambiguities.

### ✔ A **symbolic verification pipeline**  
Anyone (or any LLM) can run the checks and confirm:

\[
S = \frac{\pi k_B}{\hbar} X, 
\quad
A = k_{\mathrm{SEG}} X,
\quad
T t^\* = \Theta,
\quad
\Theta = \frac{\hbar}{\pi k_B}.
\]

### ✔ Horizon mechanics checks  
\[
X = \frac{A}{k_{\mathrm{SEG}}}, \qquad
S_{\mathrm{BH}} = \frac{k_B A}{4 \ell_P^2}.
\]

### ✔ Tick-sector checks  
\[
t^\*(T)=\frac{\hbar}{\pi k_B T}, \qquad
\tau_{\min} = \frac{t^\*}{4\pi}.
\]

### ✔ Selection & Iyer–Wald integrability  
Shows why FH selects Einstein–Hilbert uniquely.

### ✔ FH Corollaries  
Derived (not postulated) mass scales, dark-energy ratio, inertia identity, etc.

---

# 🔧 Requirements

You do **not** need to install anything if running in Colab.

Locally:

Python ≥ 3.8 is recommended.

---

# 📘 Coming Soon

- `FH_corollaries.ipynb` — numerical evaluations of FH predictions  
- `FH_playground.ipynb` — interactive calculator (ticks, BH parameters, de Sitter, etc.)

---

# 📣 Contributions

Physicists, students, and AI researchers are welcome to suggest improvements or request additional notebooks.

---

# © Author

**Enzo Cabrera Iglesias** (2025)  
Flux Holography — a constants-explicit thermodynamic formulation of GR.


