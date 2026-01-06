"""
Ivt-Hairpinstat: DNA hairpin melting temperature prediction engine.

Based on dna24 Rich Parameter model (1,334 features) for accurate Tm prediction
of DNA hairpin structures. Trained on high-throughput array experimental data.

NOTE: This model is optimized for HAIRPIN structures only, not duplexes.
For duplex Tm prediction, use primer3-py instead.

Developed by Invirustech for diagnostic kit design optimization.
"""

__version__ = "0.2.0"
__author__ = "Invirustech"


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "calculate_tm":
        from ivt_hairpinstat.core.thermodynamics import calculate_tm

        return calculate_tm
    elif name == "calculate_dG":
        from ivt_hairpinstat.core.thermodynamics import calculate_dG

        return calculate_dG
    elif name == "calculate_dH":
        from ivt_hairpinstat.core.thermodynamics import calculate_dH

        return calculate_dH
    elif name == "HairpinPredictor":
        from ivt_hairpinstat.core.predictor import HairpinPredictor

        return HairpinPredictor
    # Keep ThermoPredictor for backwards compatibility
    elif name == "ThermoPredictor":
        from ivt_hairpinstat.core.predictor import HairpinPredictor

        return HairpinPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "calculate_tm",
    "calculate_dG",
    "calculate_dH",
    "HairpinPredictor",
    "ThermoPredictor",  # backwards compatibility
]
