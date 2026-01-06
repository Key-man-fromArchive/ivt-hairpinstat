#!/usr/bin/env python3
"""
Benchmark: ivt-hairpinstat (dna24) vs ViennaRNA for DNA hairpin Tm prediction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

PAPER_DATA_DIR = Path(__file__).parent.parent.parent / "GreenleafLab-nnn_paper-802cd63" / "data"

import RNA


def calc_tm_vienna_dna(seq: str, struct: str, na_m: float = 1.0, dna_m: float = 1e-6) -> tuple:
    """
    Calculate Tm using ViennaRNA with DNA parameters.
    Returns (Tm, dH, dG_37)
    """
    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2

    fc = RNA.fold_compound(seq, md)

    dG_37 = fc.eval_structure(struct) / 100.0  # ViennaRNA returns in 10cal/mol

    # Get dH by calculating at two temperatures
    md2 = RNA.md()
    md2.temperature = 60.0
    md2.dangles = 2
    fc2 = RNA.fold_compound(seq, md2)
    dG_60 = fc2.eval_structure(struct) / 100.0

    # dG = dH - T*dS, so dH = (T2*dG1 - T1*dG2) / (T2 - T1)
    T1, T2 = 310.15, 333.15  # K
    dH = (T2 * dG_37 - T1 * dG_60) / (T2 - T1)

    # Tm for hairpin: Tm = dH / (dS + R*ln(Ct))
    # For intramolecular: Tm = dH*1000 / dS - 273.15, where dS = (dH - dG) / T
    if abs(dH) < 0.1:
        return np.nan, dH, dG_37

    dS = (dH - dG_37) / (273.15 + 37) * 1000  # cal/(mol*K)

    # Salt correction from 1M to target
    n = len(seq)
    dS_salt = dS + 0.368 * n * np.log(na_m)

    if abs(dS_salt) < 0.1:
        return np.nan, dH, dG_37

    Tm = (dH * 1000) / dS_salt - 273.15

    return Tm, dH, dG_37


def calc_tm_dna24(seq: str, struct: str, coef_dH: dict, coef_dG: dict, na_m: float = 1.0) -> tuple:
    """
    Calculate Tm using dna24 model for hairpin.
    """
    from ivt_thermostat.core.predictor import extract_features

    try:
        features = extract_features(seq, struct)
    except Exception:
        return np.nan, np.nan, np.nan

    dH = sum(coef_dH.get(f, 0.0) for f in features)
    dG_37 = sum(coef_dG.get(f, 0.0) for f in features)

    if abs(dH) < 0.1:
        return np.nan, dH, dG_37

    # Hairpin Tm: Tm = T_ref / (1 - dG/dH) - 273.15
    ratio = dG_37 / dH
    if ratio >= 1.0:
        return np.nan, dH, dG_37

    T_ref = 273.15 + 37
    Tm = T_ref / (1 - ratio) - 273.15

    # Salt correction if not 1M
    if na_m != 1.0:
        n = len(seq)
        gc = (seq.upper().count("G") + seq.upper().count("C")) / n * 100
        # Owczarzy salt correction
        ln_na = np.log(na_m)
        Tm = Tm + (4.29 * gc / 100 - 3.95) * ln_na + 9.40e-6 * (ln_na**2) * (Tm + 273.15)

    return Tm, dH, dG_37


def load_dna24_coefficients():
    import json

    coef_file = Path(__file__).parent.parent / "data" / "coefficients" / "dna24_coefficients.json"
    with open(coef_file) as f:
        data = json.load(f)

    dH_data = data.get("dH", {})
    dG_data = data.get("dG", {})
    coef_dH = dH_data.get("coefficients", dH_data) if isinstance(dH_data, dict) else {}
    coef_dG = dG_data.get("coefficients", dG_data) if isinstance(dG_data, dict) else {}
    return coef_dH, coef_dG


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    if valid.sum() == 0:
        return {"n": 0}

    actual_v, predicted_v = actual[valid], predicted[valid]
    errors = predicted_v - actual_v
    abs_errors = np.abs(errors)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actual_v - actual_v.mean()) ** 2)

    return {
        "n": len(errors),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(abs_errors),
        "bias": np.mean(errors),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0,
        "within_1": np.mean(abs_errors <= 1.0) * 100,
        "within_2": np.mean(abs_errors <= 2.0) * 100,
        "within_5": np.mean(abs_errors <= 5.0) * 100,
    }


def main():
    print("=" * 70)
    print("Hairpin Tm Benchmark: ivt-hairpinstat (dna24) vs ViennaRNA")
    print("=" * 70)

    # Load data
    data_file = PAPER_DATA_DIR / "models" / "processed" / "arr_v1_1M_n=27732.csv"
    df = pd.read_csv(data_file, index_col=0)
    print(f"\nLoaded {len(df)} hairpin samples")

    # Load dna24 coefficients
    coef_dH, coef_dG = load_dna24_coefficients()
    print(f"Loaded dna24: {len(coef_dH)} dH, {len(coef_dG)} dG features")

    # Sample for faster testing
    np.random.seed(42)
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size)
    print(f"Testing on {sample_size} samples")

    results = {
        "actual_Tm": [],
        "actual_dH": [],
        "actual_dG": [],
        "vienna_Tm": [],
        "vienna_dH": [],
        "vienna_dG": [],
        "dna24_Tm": [],
        "dna24_dH": [],
        "dna24_dG": [],
        "seq": [],
        "struct": [],
        "series": [],
    }

    for idx, row in df_sample.iterrows():
        seq = row["RefSeq"]
        struct = row["TargetStruct"]

        results["seq"].append(seq)
        results["struct"].append(struct)
        results["series"].append(row.get("Series", ""))
        results["actual_Tm"].append(row["Tm"])
        results["actual_dH"].append(row["dH"])
        results["actual_dG"].append(row["dG_37"])

        # ViennaRNA
        try:
            tm_v, dH_v, dG_v = calc_tm_vienna_dna(seq, struct)
            results["vienna_Tm"].append(tm_v)
            results["vienna_dH"].append(dH_v)
            results["vienna_dG"].append(dG_v)
        except Exception:
            results["vienna_Tm"].append(np.nan)
            results["vienna_dH"].append(np.nan)
            results["vienna_dG"].append(np.nan)

        # dna24
        try:
            tm_d, dH_d, dG_d = calc_tm_dna24(seq, struct, coef_dH, coef_dG)
            results["dna24_Tm"].append(tm_d)
            results["dna24_dH"].append(dH_d)
            results["dna24_dG"].append(dG_d)
        except Exception:
            results["dna24_Tm"].append(np.nan)
            results["dna24_dH"].append(np.nan)
            results["dna24_dG"].append(np.nan)

    # Convert to arrays
    for k in results:
        if k not in ["seq", "struct", "series"]:
            results[k] = np.array(results[k], dtype=float)

    actual_Tm = results["actual_Tm"]
    actual_dH = results["actual_dH"]
    actual_dG = results["actual_dG"]

    # Overall Tm comparison
    print("\n" + "=" * 70)
    print("Tm PREDICTION ACCURACY")
    print("=" * 70)

    for name, pred_Tm in [
        ("ViennaRNA", results["vienna_Tm"]),
        ("dna24 (ivt-hairpinstat)", results["dna24_Tm"]),
    ]:
        m = evaluate(actual_Tm, pred_Tm)
        if m["n"] > 0:
            print(f"\n{name}:")
            print(f"  N samples:    {m['n']}")
            print(f"  RMSE:         {m['rmse']:.2f}°C")
            print(f"  MAE:          {m['mae']:.2f}°C")
            print(f"  Bias:         {m['bias']:+.2f}°C")
            print(f"  R²:           {m['r2']:.4f}")
            print(f"  Within 1°C:   {m['within_1']:.1f}%")
            print(f"  Within 2°C:   {m['within_2']:.1f}%")
            print(f"  Within 5°C:   {m['within_5']:.1f}%")

    # dH comparison
    print("\n" + "=" * 70)
    print("dH PREDICTION ACCURACY")
    print("=" * 70)

    for name, pred_dH in [("ViennaRNA", results["vienna_dH"]), ("dna24", results["dna24_dH"])]:
        m = evaluate(actual_dH, pred_dH)
        if m["n"] > 0:
            print(f"\n{name}:")
            print(f"  RMSE: {m['rmse']:.2f} kcal/mol, MAE: {m['mae']:.2f}, R²: {m['r2']:.4f}")

    # dG comparison
    print("\n" + "=" * 70)
    print("dG (37°C) PREDICTION ACCURACY")
    print("=" * 70)

    for name, pred_dG in [("ViennaRNA", results["vienna_dG"]), ("dna24", results["dna24_dG"])]:
        m = evaluate(actual_dG, pred_dG)
        if m["n"] > 0:
            print(f"\n{name}:")
            print(f"  RMSE: {m['rmse']:.2f} kcal/mol, MAE: {m['mae']:.2f}, R²: {m['r2']:.4f}")

    # By structure type
    print("\n" + "=" * 70)
    print("Tm BY STRUCTURE TYPE")
    print("=" * 70)

    series_list = results["series"]
    unique_series = list(set(s for s in series_list if s))[:5]

    for series in unique_series:
        mask = np.array([s == series for s in series_list])
        n = mask.sum()
        if n < 10:
            continue

        print(f"\n--- {series} (n={n}) ---")
        for name, pred in [("ViennaRNA", results["vienna_Tm"]), ("dna24", results["dna24_Tm"])]:
            m = evaluate(actual_Tm[mask], pred[mask])
            if m["n"] > 0:
                print(
                    f"  {name:<12}: RMSE={m['rmse']:.2f}°C, R²={m['r2']:.3f}, <2°C={m['within_2']:.1f}%"
                )

    # Sample results
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    print(f"\n{'Sequence':<20} {'Struct':<18} {'Actual':>7} {'Vienna':>7} {'dna24':>7}")
    print("-" * 70)

    for i in range(min(15, len(results["seq"]))):
        seq = results["seq"][i][:18]
        struct = results["struct"][i][:16]
        actual = results["actual_Tm"][i]
        vienna = results["vienna_Tm"][i]
        dna24 = results["dna24_Tm"][i]

        v_str = f"{vienna:.1f}" if not np.isnan(vienna) else "N/A"
        d_str = f"{dna24:.1f}" if not np.isnan(dna24) else "N/A"

        print(f"{seq:<20} {struct:<18} {actual:>7.1f} {v_str:>7} {d_str:>7}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    m_vienna = evaluate(actual_Tm, results["vienna_Tm"])
    m_dna24 = evaluate(actual_Tm, results["dna24_Tm"])

    print(f"\n{'Method':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'<1°C':>8} {'<2°C':>8}")
    print("-" * 70)
    if m_vienna["n"] > 0:
        print(
            f"{'ViennaRNA':<25} {m_vienna['rmse']:>7.2f}° {m_vienna['mae']:>7.2f}° "
            f"{m_vienna['r2']:>8.4f} {m_vienna['within_1']:>7.1f}% {m_vienna['within_2']:>7.1f}%"
        )
    if m_dna24["n"] > 0:
        print(
            f"{'dna24 (ivt-hairpinstat)':<25} {m_dna24['rmse']:>7.2f}° {m_dna24['mae']:>7.2f}° "
            f"{m_dna24['r2']:>8.4f} {m_dna24['within_1']:>7.1f}% {m_dna24['within_2']:>7.1f}%"
        )

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
