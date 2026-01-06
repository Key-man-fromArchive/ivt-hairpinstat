#!/usr/bin/env python3
"""
Random oligo Tm comparison: primer3-py vs Biopython vs Manual SantaLucia vs dna24
Generates 100 random oligos (20-25mer) and compares Tm predictions.
"""

import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import primer3
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq


def generate_random_oligo(min_len: int = 20, max_len: int = 25) -> str:
    """Generate a random DNA oligo."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices("ATGC", k=length))


def calc_gc_content(seq: str) -> float:
    """Calculate GC content."""
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return (gc / len(seq)) * 100


def calc_tm_primer3(seq: str, na_mm: float = 50.0, dna_nm: float = 250.0) -> float:
    """Primer3-py (SantaLucia 1998)."""
    return primer3.calc_tm(
        seq,
        mv_conc=na_mm,
        dv_conc=0.0,
        dntp_conc=0.0,
        dna_conc=dna_nm,
        tm_method="santalucia",
        salt_corrections_method="santalucia",
    )


def calc_tm_biopython(seq: str, na_mm: float = 50.0, dna_nm: float = 250.0) -> float:
    """Biopython Tm_NN (SantaLucia 1998)."""
    return mt.Tm_NN(
        Seq(seq),
        Na=na_mm,
        dnac1=dna_nm / 2,
        dnac2=dna_nm / 2,
        nn_table=mt.DNA_NN4,
        saltcorr=5,  # SantaLucia 1998
    )


SANTALUCIA_NN = {
    "AA": (-7.9, -22.2),
    "TT": (-7.9, -22.2),
    "AT": (-7.2, -20.4),
    "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7),
    "TG": (-8.5, -22.7),
    "GT": (-8.4, -22.4),
    "AC": (-8.4, -22.4),
    "CT": (-7.8, -21.0),
    "AG": (-7.8, -21.0),
    "GA": (-8.2, -22.2),
    "TC": (-8.2, -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
    "CC": (-8.0, -19.9),
}


def calc_tm_santalucia(seq: str, na_m: float = 0.05, dna_m: float = 0.25e-6) -> float:
    """Manual SantaLucia 1998."""
    seq = seq.upper()

    dH = 0.2
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
    for i in range(len(seq) - 1):
        dinuc = seq[i : i + 2]
        if dinuc in SANTALUCIA_NN:
            dH += SANTALUCIA_NN[dinuc][0]
            dS += SANTALUCIA_NN[dinuc][1]

    # Salt correction
    dS_salt = dS + 0.368 * len(seq) * np.log(na_m)

    R = 1.987
    Ct = dna_m / 4
    Tm = (dH * 1000) / (dS_salt + R * np.log(Ct)) - 273.15

    return Tm


def extract_duplex_features_for_dna24(seq: str) -> list[str]:
    """
    Extract NN features for duplex in dna24 format.
    dna24 uses format like 'AA+TT_((+))' for NN stacks.
    """
    seq = seq.upper()
    features = []

    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}

    # Terminal feature with x/y padding
    features.append(f"x{seq[0]}+{comp[seq[0]]}y_((+))")

    # NN stacks: format is 'AB+CD_((+))' where AB is 5'->3' and CD is 3'->5' complement
    for i in range(len(seq) - 1):
        b1 = seq[i]
        b2 = seq[i + 1]
        c1 = comp[b1]
        c2 = comp[b2]
        # Feature format: 5' dinuc + 3' dinuc (reverse complement direction)
        features.append(f"{b1}{b2}+{c2}{c1}_((+))")

    # 3' terminal
    features.append(f"{seq[-1]}y+x{comp[seq[-1]]}_((+))")

    return features


def calc_tm_dna24(
    seq: str, na_m: float = 0.05, dna_m: float = 0.25e-6, coef_dH: dict = None, coef_dG: dict = None
) -> tuple[float, float, float]:
    """
    Calculate Tm using dna24 model coefficients for duplex.
    dna24 trained at 1M Na+. Returns (Tm, dH, dG_37)
    """
    if coef_dH is None or coef_dG is None:
        return np.nan, np.nan, np.nan

    features = extract_duplex_features_for_dna24(seq)

    dH = sum(coef_dH.get(f, 0.0) for f in features)
    dG_37 = sum(coef_dG.get(f, 0.0) for f in features)

    if abs(dH) < 0.1:
        return np.nan, dH, dG_37

    # dna24 dG is at 1M Na+, so dS_1M = (dH - dG) / T_ref
    T_ref = 273.15 + 37  # K
    dS_1M = (dH - dG_37) / T_ref * 1000  # cal/(mol*K)

    # Salt correction from 1M reference to target Na+ (SantaLucia 1998)
    n = len(seq)
    dS_salt = dS_1M + 0.368 * n * np.log(na_m)

    # Tm for duplex
    R = 1.987  # gas constant cal/(mol*K)
    Ct = dna_m / 4  # non-self-complementary

    Tm = (dH * 1000) / (dS_salt + R * np.log(Ct)) - 273.15

    return Tm, dH, dG_37


def load_dna24_coefficients():
    """Load dna24 model coefficients."""
    import json

    coef_file = Path(__file__).parent.parent / "data" / "coefficients" / "dna24_coefficients.json"
    if not coef_file.exists():
        return None, None

    with open(coef_file) as f:
        data = json.load(f)

    dH_data = data.get("dH", {})
    dG_data = data.get("dG", {})

    coef_dH = dH_data.get("coefficients", dH_data) if isinstance(dH_data, dict) else {}
    coef_dG = dG_data.get("coefficients", dG_data) if isinstance(dG_data, dict) else {}

    return coef_dH, coef_dG


def main():
    random.seed(42)

    print("=" * 80)
    print("Random Oligo Tm Comparison (20-25mer)")
    print("Methods: primer3-py, Biopython, Manual SantaLucia, dna24")
    print("Conditions: 50mM Na+, 250nM oligo")
    print("=" * 80)

    # Load dna24 coefficients
    coef_dH, coef_dG = load_dna24_coefficients()
    if coef_dH:
        print(f"Loaded dna24 coefficients: {len(coef_dH)} dH, {len(coef_dG)} dG features")
    else:
        print("Warning: Could not load dna24 coefficients")

    # Generate 100 random oligos
    n_oligos = 100
    oligos = [generate_random_oligo(20, 25) for _ in range(n_oligos)]

    # Standard conditions
    na_m = 0.05  # 50 mM
    dna_m = 0.25e-6  # 250 nM

    results = []
    for seq in oligos:
        gc = calc_gc_content(seq)
        tm_p3 = calc_tm_primer3(seq, na_mm=na_m * 1000, dna_nm=dna_m * 1e9)
        tm_bio = calc_tm_biopython(seq, na_mm=na_m * 1000, dna_nm=dna_m * 1e9)
        tm_sl = calc_tm_santalucia(seq, na_m=na_m, dna_m=dna_m)
        tm_dna24, dH_dna24, dG_dna24 = calc_tm_dna24(
            seq, na_m=na_m, dna_m=dna_m, coef_dH=coef_dH, coef_dG=coef_dG
        )

        results.append(
            {
                "Sequence": seq,
                "Length": len(seq),
                "GC%": gc,
                "Primer3": tm_p3,
                "Biopython": tm_bio,
                "SantaLucia": tm_sl,
                "dna24": tm_dna24,
                "dna24_dH": dH_dna24,
                "dna24_dG": dG_dna24,
            }
        )

    df = pd.DataFrame(results)

    print(f"\nGenerated {n_oligos} random oligos")
    print(f"Length range: {df['Length'].min()}-{df['Length'].max()} bp")
    print(f"GC% range: {df['GC%'].min():.1f}%-{df['GC%'].max():.1f}%")

    print("\n" + "=" * 80)
    print("Tm STATISTICS")
    print("=" * 80)

    for method in ["Primer3", "Biopython", "SantaLucia", "dna24"]:
        vals = df[method].dropna()
        print(f"\n{method}:")
        print(f"  Mean:   {vals.mean():.2f}°C")
        print(f"  Std:    {vals.std():.2f}°C")
        print(f"  Min:    {vals.min():.2f}°C")
        print(f"  Max:    {vals.max():.2f}°C")

    print("\n" + "=" * 80)
    print("METHOD COMPARISON (Primer3 as reference)")
    print("=" * 80)

    df["Bio_diff"] = df["Biopython"] - df["Primer3"]
    df["SL_diff"] = df["SantaLucia"] - df["Primer3"]
    df["dna24_diff"] = df["dna24"] - df["Primer3"]

    for name, col in [
        ("Biopython", "Bio_diff"),
        ("SantaLucia", "SL_diff"),
        ("dna24", "dna24_diff"),
    ]:
        diff = df[col].dropna()
        corr = df["Primer3"].corr(df[name.split("_")[0] if "_" in name else name])
        print(f"\n{name} vs Primer3:")
        print(f"  Mean diff:   {diff.mean():+.2f}°C")
        print(f"  Std diff:    {diff.std():.2f}°C")
        print(f"  Max diff:    {diff.abs().max():.2f}°C")
        print(f"  Correlation: {corr:.4f}")

    print("\n" + "=" * 80)
    print("SAMPLE RESULTS (first 20 oligos)")
    print("=" * 80)
    print(
        f"\n{'Sequence':<26} {'Len':>3} {'GC%':>5} {'Primer3':>8} {'Biopython':>9} {'SantaLucia':>10} {'dna24':>8}"
    )
    print("-" * 85)

    for _, row in df.head(20).iterrows():
        dna24_val = f"{row['dna24']:.2f}" if not np.isnan(row["dna24"]) else "N/A"
        print(
            f"{row['Sequence']:<26} {row['Length']:>3} {row['GC%']:>5.1f} "
            f"{row['Primer3']:>8.2f} {row['Biopython']:>9.2f} {row['SantaLucia']:>10.2f} {dna24_val:>8}"
        )

    output_file = "random_oligo_tm_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Tm BY GC% RANGE")
    print("=" * 80)

    gc_bins = [(20, 40), (40, 50), (50, 60), (60, 80)]
    print(
        f"\n{'GC% Range':<12} {'N':>5} {'Primer3':>10} {'Biopython':>10} {'SantaLucia':>10} {'dna24':>10}"
    )
    print("-" * 65)

    for low, high in gc_bins:
        mask = (df["GC%"] >= low) & (df["GC%"] < high)
        n = mask.sum()
        if n > 0:
            p3 = df.loc[mask, "Primer3"].mean()
            bio = df.loc[mask, "Biopython"].mean()
            sl = df.loc[mask, "SantaLucia"].mean()
            dna24_tm = df.loc[mask, "dna24"].mean()
            print(
                f"{low}-{high}%       {n:>5} {p3:>10.2f} {bio:>10.2f} {sl:>10.2f} {dna24_tm:>10.2f}"
            )

    print("\n" + "=" * 80)
    print("dna24 THERMODYNAMIC VALUES (sample)")
    print("=" * 80)
    print(f"\n{'Sequence':<26} {'dH (kcal/mol)':>14} {'dG (kcal/mol)':>14} {'Tm (°C)':>10}")
    print("-" * 70)
    for _, row in df.head(10).iterrows():
        if not np.isnan(row["dna24"]):
            print(
                f"{row['Sequence']:<26} {row['dna24_dH']:>14.2f} {row['dna24_dG']:>14.2f} {row['dna24']:>10.2f}"
            )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    for method in ["Primer3", "Biopython", "SantaLucia"]:
        vals = df[method]
        print(f"\n{method}:")
        print(f"  Mean:   {vals.mean():.2f}°C")
        print(f"  Std:    {vals.std():.2f}°C")
        print(f"  Min:    {vals.min():.2f}°C")
        print(f"  Max:    {vals.max():.2f}°C")

    # Method comparison
    print("\n" + "=" * 80)
    print("METHOD COMPARISON (Primer3 as reference)")
    print("=" * 80)

    df["Bio_diff"] = df["Biopython"] - df["Primer3"]
    df["SL_diff"] = df["SantaLucia"] - df["Primer3"]

    print(f"\nBiopython vs Primer3:")
    print(f"  Mean diff:  {df['Bio_diff'].mean():+.2f}°C")
    print(f"  Std diff:   {df['Bio_diff'].std():.2f}°C")
    print(f"  Max diff:   {df['Bio_diff'].abs().max():.2f}°C")
    print(f"  Correlation: {df['Primer3'].corr(df['Biopython']):.4f}")

    print(f"\nSantaLucia vs Primer3:")
    print(f"  Mean diff:  {df['SL_diff'].mean():+.2f}°C")
    print(f"  Std diff:   {df['SL_diff'].std():.2f}°C")
    print(f"  Max diff:   {df['SL_diff'].abs().max():.2f}°C")
    print(f"  Correlation: {df['Primer3'].corr(df['SantaLucia']):.4f}")

    # Sample results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS (first 20 oligos)")
    print("=" * 80)
    print(
        f"\n{'Sequence':<28} {'Len':>3} {'GC%':>5} {'Primer3':>8} {'Biopython':>9} {'SantaLucia':>10}"
    )
    print("-" * 80)

    for _, row in df.head(20).iterrows():
        print(
            f"{row['Sequence']:<28} {row['Length']:>3} {row['GC%']:>5.1f} "
            f"{row['Primer3']:>8.2f} {row['Biopython']:>9.2f} {row['SantaLucia']:>10.2f}"
        )

    # Save full results
    output_file = "random_oligo_tm_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    # Correlation by GC%
    print("\n" + "=" * 80)
    print("Tm BY GC% RANGE")
    print("=" * 80)

    gc_bins = [(20, 40), (40, 50), (50, 60), (60, 80)]
    print(f"\n{'GC% Range':<12} {'N':>5} {'Primer3':>10} {'Biopython':>10} {'SantaLucia':>10}")
    print("-" * 55)

    for low, high in gc_bins:
        mask = (df["GC%"] >= low) & (df["GC%"] < high)
        n = mask.sum()
        if n > 0:
            p3 = df.loc[mask, "Primer3"].mean()
            bio = df.loc[mask, "Biopython"].mean()
            sl = df.loc[mask, "SantaLucia"].mean()
            print(f"{low}-{high}%       {n:>5} {p3:>10.2f} {bio:>10.2f} {sl:>10.2f}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
