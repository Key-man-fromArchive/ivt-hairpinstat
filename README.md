# ivt-hairpinstat

DNA hairpin melting temperature (Tm) prediction tool based on the **dna24** paper (*Nature Communications* 2025). Trained on 27,732 unique hairpin structures from high-throughput microfluidic array experiments.

## Available Models

| Model | RMSE | R² | <2°C | Requirements |
|-------|------|-----|------|--------------|
| **GNN** | 2.74°C | 0.936 | 60% | torch, torch_geometric |
| Rich Parameter | 4.65°C | 0.815 | 50% | numpy only |

## Installation

```bash
cd /mnt/ivt-ngs1/4.paper/dna24/ivt-hairpinstat
pip install -e .

# For GNN support (optional):
pip install torch torch_geometric
```

## Quick Start

### CLI

```bash
# Default (Rich Parameter model)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))"

# GNN model (higher accuracy)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" --gnn

# Auto model selection (recommended)
ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))" --auto

# Batch prediction with auto model
ivt-hairpinstat batch sequences.csv -o results.csv --auto
```

### Python API

```python
from ivt_hairpinstat import HairpinPredictor

# Auto model selection (recommended)
predictor = HairpinPredictor(auto_select_model=True)
result = predictor.predict('GCGCAAAAGCGC', '((((....))))')
print(f"Tm: {result.Tm:.1f}°C, method: {result.prediction_method}")

# Batch prediction (3-4x faster with GPU batching)
results = predictor.predict_batch(sequences, structures, batch_size=64)

# Force specific model
predictor_gnn = HairpinPredictor(use_gnn=True)  # Always GNN
predictor_linear = HairpinPredictor()  # Always linear
```

## CLI Options

### `predict`
```
ivt-hairpinstat predict SEQUENCE [OPTIONS]

Options:
  -s, --structure     Dot-bracket structure
  -na, --sodium       Sodium concentration in M (default: 0.05)
  -g, --gnn           Use GNN model (~2.7°C RMSE)
  -a, --auto          Auto-select model (Linear for tri/tetra, GNN otherwise)
  -e, --ensemble      Ensemble prediction (weighted average of Linear + GNN)
  -t, --thermo        Use thermodynamic Tm (dH/dG)
  -j, --json          Output as JSON
```

### `batch`
```
ivt-hairpinstat batch INPUT_FILE [OPTIONS]

Options:
  -o, --output        Output CSV file
  --seq-col           Sequence column name (default: sequence)
  --struct-col        Structure column name
  -g, --gnn           Use GNN for all sequences
  -a, --auto          Auto-select model per structure type
  -b, --batch-size    Batch size for GNN (default: 64)
```

### `info`
Show available models and their status.

## Auto Model Selection

`--auto` flag automatically selects the best model per structure:

| Structure | Model Used | Why |
|-----------|------------|-----|
| Triloop (3nt) | Linear | 1.60°C RMSE vs 2.97°C |
| Tetraloop (4nt) | Linear | 2.03°C RMSE vs 2.47°C |
| Other | GNN | Better for complex structures |

This gives optimal accuracy without manual model switching.

## Ensemble Prediction

`--ensemble` / `-e` combines both models with structure-dependent weights:

```python
predictor = HairpinPredictor(auto_select_model=True)
result = predictor.predict_ensemble(sequence, structure)

# Custom weights (linear_weight, gnn_weight)
result = predictor.predict_ensemble(sequence, structure, weights=(0.5, 0.5))
```

Default weights:
- Triloop/Tetraloop: 70% Linear, 30% GNN
- Other structures: 30% Linear, 70% GNN

## Benchmark Results (2000 samples)

### Overall

| Model | RMSE | MAE | R² | <2°C | <5°C |
|-------|------|-----|-----|------|------|
| **GNN** | **2.74°C** | **2.04°C** | **0.936** | **60.1%** | **92.9%** |
| Rich Parameter | 4.65°C | 3.11°C | 0.815 | 49.9% | 80.8% |

### By Structure Type

| Type | Model | RMSE | <2°C |
|------|-------|------|------|
| TRIloop | Linear | **1.60°C** | **81%** |
| TRIloop | GNN | 2.97°C | 53% |
| TETRAloop | Linear | **2.03°C** | **75%** |
| TETRAloop | GNN | 2.47°C | 63% |
| WatsonCrick | GNN | **2.42°C** | **60%** |
| WatsonCrick | Linear | 6.90°C | 22% |
| MisMatches | GNN | **2.55°C** | **67%** |
| MisMatches | Linear | 3.78°C | 54% |
| Bulges | GNN | **3.11°C** | **51%** |
| Bulges | Linear | 6.05°C | 34% |

**Recommendation**: 
- Use **Linear** for triloops and tetraloops (structure-specific models are well-trained)
- Use **GNN** for complex structures (mismatches, bulges, Watson-Crick stems)

## Limitations

- **Hairpin Only**: For duplex Tm, use `primer3-py`
- **Reference Conditions**: 1M Na+ (salt adjustments applied via Owczarzy et al. 2004)

## Data Files

- `data/coefficients/dna24_enhanced.json`: Rich Parameter model with direct Tm
- `data/models/gnn_state_dict_ancient-sound-259.pt`: Pretrained GNN weights

## Citation

> Ke, Sharma, Wayment-Steele et al. "High-Throughput DNA melt measurements enable improved models of DNA folding thermodynamics" Nature Communications (2025)
