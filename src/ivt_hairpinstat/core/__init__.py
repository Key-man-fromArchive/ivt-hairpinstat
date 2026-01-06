__all__ = [
    "calculate_tm",
    "calculate_dG",
    "calculate_dH",
    "calculate_dS",
    "get_na_adjusted_tm",
    "HairpinPredictor",
    "ThermoPredictor",  # backwards compatibility
    "DNAStructureParser",
    "FeatureExtractor",
]


def __getattr__(name: str):
    if name in (
        "calculate_tm",
        "calculate_dG",
        "calculate_dH",
        "calculate_dS",
        "get_na_adjusted_tm",
    ):
        from ivt_hairpinstat.core import thermodynamics

        return getattr(thermodynamics, name)
    elif name in ("HairpinPredictor", "ThermoPredictor"):
        from ivt_hairpinstat.core.predictor import HairpinPredictor

        return HairpinPredictor
    elif name == "DNAStructureParser":
        from ivt_hairpinstat.core.structure_parser import DNAStructureParser

        return DNAStructureParser
    elif name == "FeatureExtractor":
        from ivt_hairpinstat.core.feature_extractor import FeatureExtractor

        return FeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
