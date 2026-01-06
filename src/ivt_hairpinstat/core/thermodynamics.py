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


def get_mg_adjusted_tm(
    Tm: float,
    GC: float,
    Mg: float,
    Na: float = 0.0,
    from_Na: float = 1.0,
    dH: Optional[float] = None,
    n_bp: Optional[int] = None,
) -> float:
    """
    Adjust melting temperature for magnesium (and optionally sodium) concentration.

    Based on Owczarzy et al. 2008 (Biochemistry 47:5336-5353).
    Handles three regimes:
    1. Mg2+ only (Na = 0)
    2. Na+ dominant (high Na/Mg ratio)
    3. Mg2+ dominant (low Na/Mg ratio)

    Args:
        Tm: Melting temperature at reference 1M Na+ (Celsius)
        GC: GC content as percentage (0-100)
        Mg: Magnesium concentration in M (e.g., 0.002 for 2mM)
        Na: Sodium concentration in M (default 0 = Mg-only)
        from_Na: Reference sodium concentration in M (default 1.0 M)
        dH: Enthalpy in kcal/mol (optional, not used in current formula)
        n_bp: Number of base pairs (optional, for length-dependent correction)

    Returns:
        Adjusted melting temperature in Celsius

    Reference:
        Owczarzy R, et al. (2008) Biochemistry 47:5336-5353
        "Predicting stability of DNA duplexes in solutions containing
        magnesium and monovalent cations"
    """
    if Mg <= 0:
        # No Mg, use Na-only correction
        if Na > 0:
            return get_na_adjusted_tm(Tm, GC, Na, from_Na, dH)
        return Tm

    Tm_kelvin = Tm + 273.15
    fGC = GC / 100.0  # Convert to fraction

    # Owczarzy 2008 coefficients for Mg correction
    a = 3.92e-5
    b = -9.11e-6
    c = 6.26e-5
    d = 1.42e-5
    e = -4.82e-4
    f = 5.25e-4
    g = 8.31e-5

    ln_Mg = np.log(Mg)

    if Na == 0:
        # Mg-only correction (Equation 16 from Owczarzy 2008)
        Tm_adj_inv = (
            1.0 / Tm_kelvin
            + a
            + b * ln_Mg
            + fGC * (c + d * ln_Mg)
            + (1.0 / (2 * (n_bp - 1)) if n_bp and n_bp > 1 else 0) * (e + f * ln_Mg + g * ln_Mg**2)
        )
    else:
        # Mixed Na/Mg conditions
        # Calculate R ratio = sqrt([Mg]) / [Na]
        R = np.sqrt(Mg) / Na

        if R < 0.22:
            # Na+ dominant - use Na-only correction
            return get_na_adjusted_tm(Tm, GC, Na, from_Na, dH)
        elif R < 6.0:
            # Intermediate regime - use modified Na correction
            # Equation 20 from Owczarzy 2008
            a_prime = 3.92e-5 * (0.843 - 0.352 * np.sqrt(Na) * ln_Mg)
            d_prime = 1.42e-5 * (1.279 - 4.03e-3 * ln_Mg - 8.03e-3 * ln_Mg**2)
            g_prime = 8.31e-5 * (0.486 - 0.258 * ln_Mg + 5.25e-3 * ln_Mg**3)

            Tm_adj_inv = (
                1.0 / Tm_kelvin
                + a_prime
                + b * ln_Mg
                + fGC * (c + d_prime * ln_Mg)
                + (1.0 / (2 * (n_bp - 1)) if n_bp and n_bp > 1 else 0)
                * (e + f * ln_Mg + g_prime * ln_Mg**2)
            )
        else:
            # Mg2+ dominant - use Mg-only correction
            Tm_adj_inv = (
                1.0 / Tm_kelvin
                + a
                + b * ln_Mg
                + fGC * (c + d * ln_Mg)
                + (1.0 / (2 * (n_bp - 1)) if n_bp and n_bp > 1 else 0)
                * (e + f * ln_Mg + g * ln_Mg**2)
            )

    return 1.0 / Tm_adj_inv - 273.15


def get_salt_adjusted_tm(
    Tm: float,
    GC: float,
    Na: float = 0.05,
    Mg: float = 0.0,
    from_Na: float = 1.0,
    dH: Optional[float] = None,
    n_bp: Optional[int] = None,
) -> float:
    """
    Unified salt correction for Tm, handling Na+, Mg2+, or both.

    This is the recommended function for salt correction as it automatically
    selects the appropriate correction method based on ion concentrations.

    Args:
        Tm: Melting temperature at reference 1M Na+ (Celsius)
        GC: GC content as percentage (0-100)
        Na: Sodium concentration in M (default 0.05 = 50mM)
        Mg: Magnesium concentration in M (default 0 = no Mg)
        from_Na: Reference sodium concentration in M (default 1.0 M)
        dH: Enthalpy in kcal/mol (optional)
        n_bp: Number of base pairs (optional)

    Returns:
        Salt-adjusted melting temperature in Celsius

    Examples:
        # Na+ only (50mM)
        >>> get_salt_adjusted_tm(70.0, 50.0, Na=0.05)

        # Mg2+ only (2mM, typical PCR)
        >>> get_salt_adjusted_tm(70.0, 50.0, Mg=0.002)

        # Mixed (50mM Na+, 2mM Mg2+)
        >>> get_salt_adjusted_tm(70.0, 50.0, Na=0.05, Mg=0.002)
    """
    if Mg > 0:
        return get_mg_adjusted_tm(Tm, GC, Mg, Na, from_Na, dH, n_bp)
    else:
        return get_na_adjusted_tm(Tm, GC, Na, from_Na, dH)


def get_salt_adjusted_dG(
    Tm: float,
    dH: float,
    GC: float,
    celsius: float = 37.0,
    Na: float = 0.05,
    Mg: float = 0.0,
    from_Na: float = 1.0,
    n_bp: Optional[int] = None,
) -> float:
    """
    Calculate salt-adjusted free energy at given temperature.

    Supports both Na+ and Mg2+ corrections.

    Args:
        Tm: Melting temperature at reference Na+ (Celsius)
        dH: Enthalpy in kcal/mol
        GC: GC content as percentage (0-100)
        celsius: Target temperature in Celsius
        Na: Sodium concentration in M
        Mg: Magnesium concentration in M
        from_Na: Reference sodium concentration in M
        n_bp: Number of base pairs (optional)

    Returns:
        Salt-adjusted free energy in kcal/mol
    """
    Tm_adjusted = get_salt_adjusted_tm(Tm, GC, Na, Mg, from_Na, dH, n_bp)
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
