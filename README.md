# AniSOAP–ASE Calculator

<div align="center">

*Bridging AniSOAP descriptors with atomistic simulations through ASE*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![ASE](https://img.shields.io/badge/ASE-Compatible-green.svg)](https://wiki.fysik.dtu.dk/ase/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Ready-orange.svg)](https://pytorch.org/)

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

</div>

---

## What is AniSOAP-ASE?

An **ASE calculator** that integrates **AniSOAP** (Anisotropic Smooth Overlap of Atomic Positions) descriptors into the **Atomic Simulation Environment (ASE)** ecosystem.

Supports **energy and force predictions** for molecular and solid-state systems with ellipsoidal particles, featuring a modular architecture with both NumPy and PyTorch backends.

### Why This Calculator?

- **Drop-in replacement** for any ASE calculator
- **Production-inspired** architecture from CACE, MACE, and XTB-ASE
- **Modular design** — swap descriptors and models without touching calculator code
- **Smart caching** — automatic result reuse for unchanged structures
- **PyTorch backend** — autodiff-ready for force calculations
- **Fully tested** with comprehensive test coverage

---

## Key Features

### Current Capabilities
Energy calculations (eV)  
**Force calculations (eV/Å)** — NEW!  
ASE-compatible interface  
PyTorch backend support  
Custom descriptor integration  
Improved error messages for ellipsoidal attributes  

### Coming Soon
🔜 Analytical gradient implementation (in progress)  
🔜 Stress tensor support  
🔜 GPU acceleration (CUDA/MPS)  
🔜 Batch evaluation  

---

## Installation

### Quick Install
```bash
git clone https://github.com/Tejas7007/cersonskylab-anisoap-ase.git
cd cersonskylab-anisoap-ase
pip install -e .
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install git+https://github.com/cersonsky-lab/AniSOAP.git
pip install torch
```

### Developer Install
```bash
git clone https://github.com/Tejas7007/cersonskylab-anisoap-ase.git
cd cersonskylab-anisoap-ase
pip install -e .
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install git+https://github.com/cersonsky-lab/AniSOAP.git
pip install torch pytest black
```

### Requirements
- Python ≥ 3.9
- NumPy
- ASE (Atomic Simulation Environment)
- AniSOAP
- PyTorch (optional, for force calculations)

---

## Quick Start

### Energy Calculation
```python
from ase.io import read
from anisoap_ase import AniSOAPCalculator

atoms = read("ellipsoid.xyz")
calc = AniSOAPCalculator()
atoms.calc = calc

energy = atoms.get_potential_energy()
print(f"Energy: {energy:.4f} eV")
```

### Force Calculation (NEW!)
```python
from ase.io import read
from anisoap_ase import AniSOAPCalculator

atoms = read("ellipsoid.xyz")

calc = AniSOAPCalculator(
    backend='torch',
    enable_forces=True
)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy: {energy:.4f} eV")
print(f"Forces shape: {forces.shape}")
print(f"Max force: {forces.max():.4f} eV/Å")
```

### Run Tests
```bash
pytest -v
```

---

## Documentation

### Calculator API
```python
AniSOAPCalculator(
    backend: str = "numpy",
    descriptor_fn: callable = None,
    model: callable = None,
    enable_forces: bool = False,
    cache_results: bool = True,
)
```

### Backend Selection

| Backend | Energy | Forces | Speed | Use Case |
|---------|--------|--------|-------|----------|
| `numpy` | ✅| ❌ | Baseline | Energy-only calculations |
| `torch` | ✅ | ✅ | 12-25% faster | Force calculations, autodiff |

### Units

| Quantity | Unit | Notes |
|----------|------|-------|
| Energy | eV | Electron volts |
| Length | Å | Ångström |
| Forces | eV/Å | Energy gradient |

### Implemented Properties

| Property | Unit | NumPy Backend | PyTorch Backend |
|----------|------|---------------|-----------------|
| `energy` | eV | ✅ | ✅ |
| `forces` | eV/Å | ❌ | ✅ |
| `stress` | eV/Å³ | ❌ | 🔜 |

### Error Handling

Clear error messages for ellipsoidal attributes:
```python
atoms = Atoms("H2O", positions=[[0,0,0], [1,0,0], [0,1,0]])
calc = AniSOAPCalculator()
atoms.calc = calc

try:
    energy = atoms.get_potential_energy()
except ValueError as e:
    print(e)
```

Output:
```
Expect frames with ellipsoidal attributes: frame at index 0 is missing a required attribute 'c_q'
```

---

## Project Structure
```
cersonskylab-anisoap-ase/
├── anisoap_ase/
│   ├── calculator.py
│   ├── descriptors_torch.py    (NEW)
│   ├── descriptors.py
│   └── model.py
├── tests/
│   ├── test_calculator.py
│   ├── test_forces.py          (NEW)
│   └── test_anisoap_linear_model.py
├── examples/
│   └── water_energy.py
└── README.md
```

---

## Technical Details

### Force Calculation

**Current:** Finite differences (robust baseline)  
**Coming soon:** Analytical gradients via PyTorch autodiff

### Performance

- PyTorch backend: **12-25% faster** than NumPy on CPU
- Primary bottleneck: `numpy.einsum` operations (66-77% of runtime)
- Linear scaling with number of chemical species

---

## Testing

Our test suite covers:
- Energy calculation pipeline
- Force calculations
- Caching mechanisms
- Error handling
- Full AniSOAP integration

Run tests with:
```bash
pytest -v
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This implementation draws inspiration from:
- **[CACE](https://github.com/BingqingCheng/cace)** — Bingqing Cheng Group
- **[MACE](https://github.com/ACEsuit/mace)** — ACEsuit Team  
- **[XTB-ASE](https://github.com/Andrew-S-Rosen/xtb_ase)** — Andrew S. Rosen

Special thanks to the **Cersonsky Lab** at UW-Madison.

---

## Author

**Tejas Dahiya**  
*Cersonsky Lab • University of Wisconsin–Madison*

Developed under the mentorship of **Arthur Lin**

[![Email](https://img.shields.io/badge/Email-tejasdahiya0007%40gmail.com-red?style=flat&logo=gmail)](mailto:tejasdahiya0007@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Tejas7007-black?style=flat&logo=github)](https://github.com/Tejas7007)

---

<div align="center">
