"""
Core thermodynamic calculation functions.

Based on dna24 paper formulas for DNA folding thermodynamics.
All calculations use kcal/mol for energy units.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Gas constant in kcal/(mol*K)
R = 0.00198717


def calculate_dG(dH: float, Tm: float, celsius: float = 37.0) -> float:
    """
    Calculate free energy (dG) at a given temperature.

    Formula: dG(T) = dH * (1 - T/Tm)

    Args:
        dH: Enthalpy in kcal/mol (negative for stable structures)
        Tm: Melting temperature in Celsius
        celsius: Target temperature in Celsius (default 37C)

    Returns:
        Free energy dG in kcal/mol
    """
    T_kelvin = 273.15 + celsius
    Tm_kelvin = 273.15 + Tm
    return dH * (1 - T_kelvin / Tm_kelvin)


def calculate_tm(dH: float, dG: float, celsius: float = 37.0) -> float:
    """
    Calculate melting temperature from dH and dG at reference temperature.

    Formula: Tm = T_ref / (1 - dG/dH) - 273.15

    Args:
        dH: Enthalpy in kcal/mol
        dG: Free energy at reference temperature in kcal/mol
        celsius: Reference temperature in Celsius (default 37C)

    Returns:
        Melting temperature in Celsius
    """
    if dH == 0:
        return np.nan
    T_ref_kelvin = 273.15 + celsius
    return T_ref_kelvin / (1 - dG / dH) - 273.15


def calculate_dS(dH: float, Tm: float) -> float:
    """
    Calculate entropy from enthalpy and melting temperature.

    Formula: dS = dH / Tm (in Kelvin)

    Args:
        dH: Enthalpy in kcal/mol
        Tm: Melting temperature in Celsius

    Returns:
        Entropy dS in kcal/(mol*K)
    """
    Tm_kelvin = Tm + 273.15
    return dH / Tm_kelvin


def calculate_dH(dG: float, Tm: float, celsius: float = 37.0) -> float:
    """
    Calculate enthalpy from free energy and melting temperature.

    Derived from: dG = dH * (1 - T/Tm)
    Therefore: dH = dG / (1 - T/Tm)

    Args:
        dG: Free energy at reference temperature in kcal/mol
        Tm: Melting temperature in Celsius
        celsius: Reference temperature in Celsius (default 37C)

    Returns:
        Enthalpy dH in kcal/mol
    """
    T_kelvin = 273.15 + celsius
    Tm_kelvin = 273.15 + Tm
    ratio = 1 - T_kelvin / Tm_kelvin
    if ratio == 0:
        return np.nan
    return dG / ratio


def get_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.

    Args:
        sequence: DNA sequence string (ATCG)

    Returns:
        GC content as percentage (0-100)
    """
    sequence = sequence.upper().replace(" ", "").replace("+", "")
    gc_count = sum(1 for base in sequence if base in "GC")
    return 100.0 * gc_count / len(sequence) if sequence else 0.0


def get_na_adjusted_tm(
    Tm: float,
    GC: float,
    Na: float = 0.063,
    from_Na: float = 1.0,
    dH: Optional[float] = None,
) -> float:
    """
    Adjust melting temperature for sodium concentration.

    Uses high-order polynomial correction from dna24 paper:
    1/Tm_adj = 1/Tm + (4.29*fGC - 3.95)*1e-5*ln(Na/Na_ref)
               + 9.4*1e-6*(ln(Na)^2 - ln(Na_ref)^2)

    Args:
        Tm: Melting temperature at reference Na+ concentration (Celsius)
        GC: GC content as percentage (0-100)
        Na: Target sodium concentration in M (default 0.063 M = 63 mM)
        from_Na: Reference sodium concentration in M (default 1.0 M)
        dH: Enthalpy (not used in current formula, kept for compatibility)

    Returns:
        Adjusted melting temperature in Celsius
    """
    Tm_kelvin = Tm + 273.15
    fGC = GC / 100.0  # Convert to fraction

    # High-order sodium correction
    ln_ratio = np.log(Na / from_Na)
    Tm_adj_inv = (
        1.0 / Tm_kelvin
        + (4.29 * fGC - 3.95) * 1e-5 * ln_ratio
        + 9.4e-6 * (np.log(Na) ** 2 - np.log(from_Na) ** 2)
    )

    return 1.0 / Tm_adj_inv - 273.15


def get_na_adjusted_dG(
    Tm: float,
    dH: float,
    GC: float,
    celsius: float = 37.0,
    Na: float = 0.063,
    from_Na: float = 1.0,
) -> float:
    """
    Calculate sodium-adjusted free energy at given temperature.

    Args:
        Tm: Melting temperature at reference Na+ (Celsius)
        dH: Enthalpy in kcal/mol
        GC: GC content as percentage (0-100)
        celsius: Target temperature in Celsius
        Na: Target sodium concentration in M
        from_Na: Reference sodium concentration in M

    Returns:
        Sodium-adjusted free energy in kcal/mol
    """
    Tm_adjusted = get_na_adjusted_tm(Tm, GC, Na, from_Na)
    return calculate_dG(dH, Tm_adjusted, celsius)


def calculate_duplex_tm(
    dH: float,
    dG: float,
    DNA_conc: float,
    is_self_complementary: bool = False,
    celsius: float = 37.0,
) -> float:
    """
    Calculate melting temperature for a DNA duplex.

    For duplexes, Tm depends on strand concentration:
    Tm = dH / (dH - dG)/T_ref - R*ln(K)) - 273.15

    where K = 1/C for self-complementary, 2/C for non-self-complementary

    Args:
        dH: Enthalpy in kcal/mol
        dG: Free energy at reference temperature in kcal/mol
        DNA_conc: Total strand concentration in M
        is_self_complementary: True if strands are identical
        celsius: Reference temperature in Celsius

    Returns:
        Melting temperature in Celsius
    """
    T_ref_kelvin = 273.15 + celsius

    # Equilibrium constant depends on whether strands are distinguishable
    if is_self_complementary:
        lnK = np.log(1.0 / DNA_conc)
    else:
        lnK = np.log(2.0 / DNA_conc)

    dS_eff = (dH - dG) / T_ref_kelvin
    Tm = dH / (dS_eff - R * lnK) - 273.15

    return Tm


def calculate_dG_error(
    dH: float,
    dH_err: float,
    Tm: float,
    Tm_err: float,
    celsius: float = 37.0,
) -> float:
    """
    Calculate error in dG using error propagation.

    dG = dH - T*dS, where dS = dH/Tm

    Args:
        dH: Enthalpy in kcal/mol
        dH_err: Standard error of dH
        Tm: Melting temperature in Celsius
        Tm_err: Standard error of Tm
        celsius: Target temperature in Celsius

    Returns:
        Standard error of dG in kcal/mol
    """
    T_kelvin = celsius + 273.15
    dS_err = calculate_dS_error(dH, dH_err, Tm, Tm_err)
    return np.sqrt(dH_err**2 + (T_kelvin * dS_err) ** 2)


def calculate_dS_error(
    dH: float,
    dH_err: float,
    Tm: float,
    Tm_err: float,
) -> float:
    """
    Calculate error in dS using error propagation.

    dS = dH / Tm
    dS_err = |dS| * sqrt((dH_err/dH)^2 + (Tm_err/Tm)^2)

    Args:
        dH: Enthalpy in kcal/mol
        dH_err: Standard error of dH
        Tm: Melting temperature in Celsius
        Tm_err: Standard error of Tm

    Returns:
        Standard error of dS in kcal/(mol*K)
    """
    dS = calculate_dS(dH, Tm)
    Tm_kelvin = Tm + 273.15
    return abs(dS) * np.sqrt((dH_err / dH) ** 2 + (Tm_err / Tm_kelvin) ** 2)
