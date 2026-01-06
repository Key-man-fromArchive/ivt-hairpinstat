# ivt-hairpinstat

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fast, accurate DNA hairpin Tm prediction without NUPACK dependency.**

Based on the **dna24** paper (*Nature Communications* 2025), trained on 27,732 unique hairpin structures from high-throughput microfluidic array experiments.

## Why ivt-hairpinstat?

| Feature | ivt-hairpinstat | NUPACK | ViennaRNA |
|---------|-----------------|--------|-----------|
| **Tm RMSE** | **2.7°C** | ~4-5°C | ~5-6°C |
| **Dependencies** | numpy only* | Complex C++ | C library |
| **Installation** | `pip install` | Build from source | System packages |
| **Python native** | ✅ | ❌ | ❌ |
| **Commercial use** | MIT License | Restricted | GPL |

*\*GNN model requires PyTorch (optional)*

### Accuracy Comparison

Tested against experimental data (n=1000 hairpins):

| Model | dG RMSE | Tm RMSE | Predictions within 2°C |
|-------|---------|---------|------------------------|
| ViennaRNA DNA | 1.45 kcal/mol | ~5-6°C | ~30% |
| RNAstructure DNA | 1.43 kcal/mol | ~5-6°C | ~30% |
| **ivt Linear** | 0.58 kcal/mol | 4.48°C | 48% |
| **ivt GNN** | **0.47 kcal/mol** | **2.71°C** | **57%** |

## Installation

```bash
# Basic installation (Linear model only - no heavy dependencies)
pip install git+https://github.com/Key-man-fromArchive/ivt-hairpinstat.git

# With GNN support (higher accuracy, requires PyTorch)
pip install "ivt-hairpinstat[gnn] @ git+https://github.com/Key-man-fromArchive/ivt-hairpinstat.git"
```

### From source
```bash
git clone https://github.com/Key-man-fromArchive/ivt-hairpinstat.git
cd ivt-hairpinstat
pip install -e .          # Linear model only
pip install -e ".[gnn]"   # With GNN support
```

## Quick Start

### Command Line

```bash
# Simple prediction (Linear model)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))"

# GNN model (higher accuracy)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" --gnn

# Auto model selection (recommended - best of both)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" --auto

# Custom sodium concentration (default: 50mM)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" --sodium 0.1

# Batch processing
ivt-hairpinstat batch sequences.csv -o results.csv --auto
```

### Python API

```python
from ivt_hairpinstat import HairpinPredictor

# Auto model selection (recommended)
predictor = HairpinPredictor(auto_select_model=True)
result = predictor.predict('GCGCAAAAGCGC', '((((....))))')
print(f"Tm: {result.Tm:.1f}°C (±{result.Tm_error:.1f}°C)")
print(f"dG: {result.dG:.2f} kcal/mol")
print(f"Method: {result.prediction_method}")

# Batch prediction (3-4x faster with GPU)
sequences = ['GCGCAAAAGCGC', 'GGGGTTTTCCCC', ...]
structures = ['((((....))))', '((((....))))', ...]
results = predictor.predict_batch(sequences, structures, batch_size=64)

# Custom conditions
result = predictor.predict(
    sequence='GCGCAAAAGCGC',
    structure='((((....))))',
    sodium=0.15,  # 150mM Na+
    celsius=37.0  # Temperature for dG calculation
)

# Ensemble prediction (weighted average of both models)
result = predictor.predict_ensemble(sequence, structure)
```

## Available Models

### Linear Model (Rich Parameter)
- **1,334 NNN thermodynamic features**
- **No heavy dependencies** - numpy only
- Best for triloops/tetraloops: **1.6-2.0°C RMSE**
- Overall: 4.48°C RMSE

### GNN Model (Graph Transformer)
- **287K parameters**, Set2Set pooling
- Requires PyTorch + PyTorch Geometric
- Best for complex structures: **2.4-3.1°C RMSE**
- Overall: **2.71°C RMSE**

### Auto Selection (`--auto`)

Automatically picks the best model per structure:

| Structure Type | Model Used | RMSE |
|----------------|------------|------|
| Triloop (3nt loop) | Linear | 1.60°C |
| Tetraloop (4nt loop) | Linear | 2.03°C |
| Mismatches | GNN | 2.55°C |
| Bulges | GNN | 3.11°C |
| Watson-Crick only | GNN | 2.42°C |

## CLI Reference

### `predict`
```
ivt-hairpinstat predict SEQUENCE [OPTIONS]

Options:
  -s, --structure TEXT    Dot-bracket structure (required)
  -na, --sodium FLOAT     Na+ concentration in M [default: 0.05]
  -mg, --magnesium FLOAT  Mg2+ concentration in M [default: 0]
  -g, --gnn               Use GNN model
  -a, --auto              Auto-select best model per structure
  -e, --ensemble          Ensemble prediction (Linear + GNN weighted)
  -t, --thermo            Show thermodynamic Tm from dH/dG
  -j, --json              Output as JSON
```

### `batch`
```
ivt-hairpinstat batch INPUT_FILE [OPTIONS]

Options:
  -o, --output TEXT       Output CSV file
  --seq-col TEXT          Sequence column [default: sequence]
  --struct-col TEXT       Structure column [default: structure]
  -na, --sodium FLOAT     Na+ concentration in M [default: 0.05]
  -mg, --magnesium FLOAT  Mg2+ concentration in M [default: 0]
  -g, --gnn               Use GNN for all
  -a, --auto              Auto-select per structure
  -b, --batch-size INT    GNN batch size [default: 64]
```

### `info`
```
ivt-hairpinstat info      # Show model status and versions
```

## Salt Correction

Supports both **Na+** and **Mg2+** corrections.

### Sodium-only (dna24 paper)
```
1/Tm_adj = 1/Tm + (4.29*fGC - 3.95)*1e-5*ln([Na+]/[Na+]_ref)
           + 9.4*1e-6*(ln([Na+])² - ln([Na+]_ref)²)
```

### Magnesium correction (Owczarzy et al. 2008)

Three regimes based on R = √[Mg²⁺]/[Na⁺]:
- **R < 0.22**: Na+ dominant → use Na-only correction
- **0.22 ≤ R < 6**: Mixed → modified correction
- **R ≥ 6**: Mg2+ dominant → use Mg-only correction

### Examples

```bash
# Na+ only (50mM, default)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))"

# Mg2+ only (2mM, typical PCR)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" -mg 0.002 -na 0

# Mixed (50mM Na+, 1.5mM Mg2+)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" -na 0.05 -mg 0.0015
```

```python
from ivt_hairpinstat import HairpinPredictor
from ivt_hairpinstat.core.predictor import SaltConditions

predictor = HairpinPredictor(auto_select_model=True)
salt = SaltConditions(Na=0.05, Mg=0.0015)  # Typical PCR buffer
result = predictor.predict('GCGCAAAAGCGC', '((((....))))', salt_conditions=salt)
print(f"Tm (adjusted): {result.Tm_adjusted:.1f}°C")
```

### Valid Ranges
- Reference: 1M Na+
- Na+: 10mM - 1M
- Mg2+: 0.5mM - 50mM

## Limitations

- **Hairpin structures only** - For duplex Tm, use [primer3-py](https://github.com/libnano/primer3-py)
- **DNA only** - RNA hairpins not supported
- **Single hairpin** - Multi-hairpin structures require manual decomposition

## Citation

If you use ivt-hairpinstat in your research, please cite:

> Ke, Sharma, Wayment-Steele et al. "High-Throughput DNA melt measurements enable improved models of DNA folding thermodynamics" *Nature Communications* (2025)

## License

MIT License - free for academic and commercial use.
