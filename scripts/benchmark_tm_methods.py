#!/usr/bin/env python3
"""
Benchmark Tm prediction: Ivt-Thermostat vs primer3-py vs Biopython

Compares prediction accuracy on experimental duplex Tm data.
Note: dna24/Ivt-Thermostat is optimized for HAIRPIN structures, not duplexes.
For duplex Tm, it uses classical nearest-neighbor (SantaLucia) approach.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

PAPER_DATA_DIR = Path(__file__).parent.parent.parent / "GreenleafLab-nnn_paper-802cd63" / "data"

import primer3
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq


def calc_tm_primer3(seq: str, na_mm: float = 50.0, dna_um: float = 0.25) -> float:
    """Calculate Tm using primer3-py (SantaLucia 1998)."""
    return primer3.calc_tm(
        seq,
        mv_conc=na_mm,
        dv_conc=0.0,
        dntp_conc=0.0,
        dna_conc=dna_um * 1000,
        tm_method="santalucia",
        salt_corrections_method="santalucia",
    )


def calc_tm_biopython_nn(seq: str, na_mm: float = 50.0, dna_um: float = 0.25) -> float:
    """Calculate Tm using Biopython nearest-neighbor (SantaLucia 1998)."""
    return mt.Tm_NN(
        Seq(seq),
        Na=na_mm,
        dnac1=dna_um * 1e6 / 2,
        dnac2=dna_um * 1e6 / 2,
        nn_table=mt.DNA_NN4,
        saltcorr=7,
    )


def calc_tm_biopython_gc(seq: str) -> float:
    """Simple %GC method (Wallace rule)."""
    return mt.Tm_Wallace(Seq(seq))


# SantaLucia 1998 unified NN parameters for duplexes
SANTALUCIA_NN = {
    "AA/TT": (-7.9, -22.2),
    "AT/TA": (-7.2, -20.4),
    "TA/AT": (-7.2, -21.3),
    "CA/GT": (-8.5, -22.7),
    "GT/CA": (-8.4, -22.4),
    "CT/GA": (-7.8, -21.0),
    "GA/CT": (-8.2, -22.2),
    "CG/GC": (-10.6, -27.2),
    "GC/CG": (-9.8, -24.4),
    "GG/CC": (-8.0, -19.9),
    "init_A/T": (2.3, 4.1),
    "init_G/C": (0.1, -2.8),
    "init_oneG/C": (0.98, -0.7),
    "init_allA/T": (1.03, 2.3),
    "init": (0.2, -5.7),  # initiation
    "sym": (0, -1.4),  # symmetry correction for self-complementary
}


def calc_tm_santalucia_manual(
    seq: str, na_m: float = 0.05, dna_m: float = 0.25e-6, self_comp: bool = False
) -> float:
    """Manual SantaLucia 1998 implementation for comparison."""
    seq = seq.upper()

    # Initialize
    dH = 0.2  # initiation
    dS = -5.7

    # Terminal penalties
    if seq[0] in "AT":
        dH += 2.3
        dS += 4.1
    else:
        dH += 0.1
        dS += -2.8

    if seq[-1] in "AT":
        dH += 2.3
        dS += 4.1
    else:
        dH += 0.1
        dS += -2.8

    # NN stacks
    nn_map = {
        "AA": ("AA/TT", 1),
        "TT": ("AA/TT", 1),
        "AT": ("AT/TA", 1),
        "TA": ("TA/AT", 1),
        "CA": ("CA/GT", 1),
        "TG": ("CA/GT", 1),
        "GT": ("GT/CA", 1),
        "AC": ("GT/CA", 1),
        "CT": ("CT/GA", 1),
        "AG": ("CT/GA", 1),
        "GA": ("GA/CT", 1),
        "TC": ("GA/CT", 1),
        "CG": ("CG/GC", 1),
        "GC": ("GC/CG", 1),
        "GG": ("GG/CC", 1),
        "CC": ("GG/CC", 1),
    }

    for i in range(len(seq) - 1):
        dinuc = seq[i : i + 2]
        if dinuc in nn_map:
            key, _ = nn_map[dinuc]
            dH += SANTALUCIA_NN[key][0]
            dS += SANTALUCIA_NN[key][1]

    # Symmetry correction
    if self_comp:
        dS += -1.4

    # Salt correction (SantaLucia 1998)
    n = len(seq)
    dS_salt = dS + 0.368 * n * np.log(na_m)

    # Tm calculation
    R = 1.987  # gas constant cal/(mol*K)
    if self_comp:
        Ct = dna_m
    else:
        Ct = dna_m / 4  # non-self-comp: Ct/4

    Tm = (dH * 1000) / (dS_salt + R * np.log(Ct)) - 273.15

    return Tm


def is_self_complementary(seq: str) -> bool:
    """Check if sequence is self-complementary."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_comp = "".join(comp.get(b, "N") for b in reversed(seq.upper()))
    return seq.upper() == rev_comp


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate error metrics."""
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() == 0:
        return {"n": 0}

    actual_v = actual[valid]
    predicted_v = predicted[valid]
    errors = predicted_v - actual_v
    abs_errors = np.abs(errors)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actual_v - actual_v.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "n": len(errors),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(abs_errors),
        "bias": np.mean(errors),
        "r2": r2,
        "within_0.5": np.mean(abs_errors <= 0.5) * 100,
        "within_1.0": np.mean(abs_errors <= 1.0) * 100,
        "within_2.0": np.mean(abs_errors <= 2.0) * 100,
        "within_5.0": np.mean(abs_errors <= 5.0) * 100,
    }


def print_metrics(name: str, metrics: dict):
    if metrics["n"] == 0:
        print(f"{name}: No valid predictions")
        return

    print(f"\n{name}")
    print("-" * 50)
    print(f"  N samples:     {metrics['n']}")
    print(f"  RMSE:          {metrics['rmse']:.2f}°C")
    print(f"  MAE:           {metrics['mae']:.2f}°C")
    print(f"  Bias:          {metrics['bias']:+.2f}°C")
    print(f"  R²:            {metrics['r2']:.4f}")
    print(f"  Within 0.5°C:  {metrics['within_0.5']:.1f}%")
    print(f"  Within 1.0°C:  {metrics['within_1.0']:.1f}%")
    print(f"  Within 2.0°C:  {metrics['within_2.0']:.1f}%")
    print(f"  Within 5.0°C:  {metrics['within_5.0']:.1f}%")


def main():
    print("=" * 70)
    print("Tm Prediction Benchmark: Duplex DNA (5-60mer)")
    print("Comparing: primer3-py, Biopython, manual SantaLucia")
    print("=" * 70)

    # Load experimental data
    lit_file = PAPER_DATA_DIR / "literature" / "compiled_DNA_Tm_348oligos.csv"
    if not lit_file.exists():
        print(f"Error: Data file not found: {lit_file}")
        sys.exit(1)

    df = pd.read_csv(lit_file, index_col=0)
    print(f"\nLoaded {len(df)} experimental Tm measurements")
    print(f"Length range: {df['Length (bp)'].min()}-{df['Length (bp)'].max()} bp")

    # Filter to oligo lengths (5-60mer)
    df = df[(df["Length (bp)"] >= 5) & (df["Length (bp)"] <= 60)].copy()
    print(f"After filtering 5-60 bp: {len(df)} samples")

    # Collect predictions
    results = {
        "actual": [],
        "primer3": [],
        "biopython_nn": [],
        "biopython_gc": [],
        "santalucia_manual": [],
    }

    for idx, row in df.iterrows():
        seq = str(row["seq"])
        actual_tm = float(row["Tm"])
        na_m = float(row["sodium"])
        dna_m = float(row["DNA_conc"])
        self_comp = is_self_complementary(seq)

        results["actual"].append(actual_tm)

        # Primer3
        try:
            tm = calc_tm_primer3(seq, na_mm=na_m * 1000, dna_um=dna_m * 1e6)
            results["primer3"].append(tm)
        except Exception:
            results["primer3"].append(np.nan)

        # Biopython NN
        try:
            tm = calc_tm_biopython_nn(seq, na_mm=na_m * 1000, dna_um=dna_m * 1e6)
            results["biopython_nn"].append(tm)
        except Exception:
            results["biopython_nn"].append(np.nan)

        # Biopython GC
        try:
            tm = calc_tm_biopython_gc(seq)
            results["biopython_gc"].append(tm)
        except Exception:
            results["biopython_gc"].append(np.nan)

        # Manual SantaLucia
        try:
            tm = calc_tm_santalucia_manual(seq, na_m=na_m, dna_m=dna_m, self_comp=self_comp)
            results["santalucia_manual"].append(tm)
        except Exception:
            results["santalucia_manual"].append(np.nan)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key], dtype=float)

    actual = results["actual"]

    # Overall results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print_metrics("Primer3-py (SantaLucia 1998)", evaluate_predictions(actual, results["primer3"]))
    print_metrics(
        "Biopython Tm_NN (SantaLucia 1998 + Owczarzy salt)",
        evaluate_predictions(actual, results["biopython_nn"]),
    )
    print_metrics(
        "Biopython Tm_Wallace (%GC rule)", evaluate_predictions(actual, results["biopython_gc"])
    )
    print_metrics(
        "Manual SantaLucia 1998", evaluate_predictions(actual, results["santalucia_manual"])
    )

    # By salt concentration
    print("\n" + "=" * 70)
    print("RESULTS BY SODIUM CONCENTRATION")
    print("=" * 70)

    na_concs = df["sodium"].values

    for label, mask in [
        ("Low Salt (<100 mM)", na_concs < 0.1),
        ("Standard (100-200 mM)", (na_concs >= 0.1) & (na_concs <= 0.2)),
        ("High Salt (>=500 mM)", na_concs >= 0.5),
    ]:
        if mask.sum() > 10:
            print(f"\n--- {label}, n={mask.sum()} ---")
            for name, pred in [
                ("Primer3", results["primer3"]),
                ("Biopython NN", results["biopython_nn"]),
                ("Manual SantaLucia", results["santalucia_manual"]),
            ]:
                m = evaluate_predictions(actual[mask], pred[mask])
                if m["n"] > 0:
                    print(
                        f"  {name:<20}: RMSE={m['rmse']:.2f}°C, <1°C={m['within_1.0']:.1f}%, R²={m['r2']:.3f}"
                    )

    # By length
    print("\n" + "=" * 70)
    print("RESULTS BY OLIGO LENGTH")
    print("=" * 70)

    lengths = df["Length (bp)"].values

    for min_len, max_len in [(5, 15), (16, 20), (21, 25), (26, 40), (41, 60)]:
        mask = (lengths >= min_len) & (lengths <= max_len)
        if mask.sum() > 5:
            print(f"\n--- {min_len}-{max_len} bp, n={mask.sum()} ---")
            for name, pred in [
                ("Primer3", results["primer3"]),
                ("Biopython NN", results["biopython_nn"]),
                ("Manual SantaLucia", results["santalucia_manual"]),
            ]:
                m = evaluate_predictions(actual[mask], pred[mask])
                if m["n"] > 0:
                    print(
                        f"  {name:<20}: RMSE={m['rmse']:.2f}°C, <1°C={m['within_1.0']:.1f}%, R²={m['r2']:.3f}"
                    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'Method':<30} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'R²':>8} {'<1°C':>8} {'<2°C':>8}")
    print("-" * 82)

    for name, pred in [
        ("Primer3-py", results["primer3"]),
        ("Biopython NN", results["biopython_nn"]),
        ("Biopython GC (Wallace)", results["biopython_gc"]),
        ("Manual SantaLucia 1998", results["santalucia_manual"]),
    ]:
        m = evaluate_predictions(actual, pred)
        if m["n"] > 0:
            print(
                f"{name:<30} {m['rmse']:>7.2f}° {m['mae']:>7.2f}° {m['bias']:>+7.2f}° {m['r2']:>8.4f} {m['within_1.0']:>7.1f}% {m['within_2.0']:>7.1f}%"
            )

    print("\n" + "=" * 70)
    print("NOTE: dna24/Ivt-Thermostat is optimized for HAIRPIN structures.")
    print("For duplex Tm, standard SantaLucia nearest-neighbor is used.")
    print("=" * 70)


if __name__ == "__main__":
    main()
