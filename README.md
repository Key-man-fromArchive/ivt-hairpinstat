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
  -g, --gnn               Use GNN for all
  -a, --auto              Auto-select per structure
  -b, --batch-size INT    GNN batch size [default: 64]
```

### `info`
```
ivt-hairpinstat info      # Show model status and versions
```

## Salt Correction

Uses high-order polynomial correction from dna24 paper:

```
1/Tm_adj = 1/Tm + (4.29*fGC - 3.95)*1e-5*ln([Na+]/[Na+]_ref)
           + 9.4*1e-6*(ln([Na+])² - ln([Na+]_ref)²)
```

- Reference condition: 1M Na+
- Valid range: 10mM - 1M Na+

**Note**: Magnesium correction is not yet implemented. For Mg2+ containing buffers, contributions from Mg2+ should be considered separately.

## Limitations

- **Hairpin structures only** - For duplex Tm, use [primer3-py](https://github.com/libnano/primer3-py)
- **DNA only** - RNA hairpins not supported
- **No Mg2+ correction** - Sodium-only salt adjustment
- **Single hairpin** - Multi-hairpin structures require manual decomposition

## Citation

If you use ivt-hairpinstat in your research, please cite:

> Ke, Sharma, Wayment-Steele et al. "High-Throughput DNA melt measurements enable improved models of DNA folding thermodynamics" *Nature Communications* (2025)

## License

MIT License - free for academic and commercial use.
