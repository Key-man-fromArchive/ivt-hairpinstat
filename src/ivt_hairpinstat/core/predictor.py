"""
Hairpin Tm predictor using dna24 models.

Main prediction engine for Ivt-Hairpinstat.

Based on dna24 paper (Nature Communications 2025) - trained on 27,732 hairpin structures.

Available models:
1. Rich Parameter (default): 1,334 NNN features for linear regression
2. GNN: Graph Neural Network with ~287K parameters (requires torch, torch_geometric)

NOTE: This model is optimized for HAIRPIN structures only, not duplexes.
For duplex Tm prediction, use primer3-py instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ivt_hairpinstat.core.thermodynamics import (
    calculate_tm,
    calculate_dG,
    get_gc_content,
    get_na_adjusted_tm,
    get_salt_adjusted_tm,
    calculate_duplex_tm,
)

HAS_RIBOGRAPHVIZ = False
_LoopExtruder = None
_StackExtruder = None

try:
    from RiboGraphViz import LoopExtruder as _LoopExtruder, StackExtruder as _StackExtruder  # type: ignore

    HAS_RIBOGRAPHVIZ = True
except ImportError:
    pass


def _clean_feature(x: str) -> str:
    """Clean feature string from RiboGraphViz format to standard format."""
    cleaned = x.replace(" ", "+").replace(",", "_")
    # Handle flipped terminal: if 'y' is in position 1 and structure is ((+))
    if len(cleaned) > 4 and cleaned[1] == "y" and cleaned.split("_")[1] == "((+))":
        # Flip terminal: 'Ay+Bx_((+))' -> 'xB+Ay_((+))'
        cleaned = f"x{cleaned[4]}+{cleaned[0]}y_((+))"
    return cleaned


def extract_features_rgv(
    seq: str,
    struct: str,
    stack_size: int = 2,
    sep_base_stack: bool = True,
    hairpin_mm: bool = False,
) -> list[str]:
    """
    Extract features from sequence and structure using RiboGraphViz.

    This matches the paper's get_feature_list() function.

    Args:
        seq: DNA sequence
        struct: Dot-bracket structure
        stack_size: Size of nearest neighbor stacks (default 2 for NN)
        sep_base_stack: Separate hairpin base stack from loop
        hairpin_mm: Add hairpin mismatch parameters

    Returns:
        List of feature strings in NNN format
    """
    hp_pattern = re.compile(r"^\([.]+\)")
    loop_base_size = 1
    pad = stack_size - 1

    # Pad sequence and structure with x/y symbols
    seq_padded = "x" * pad + seq + "y" * pad
    struct_padded = "(" * pad + struct + ")" * pad

    # Extract loops and stacks using RiboGraphViz
    assert _LoopExtruder is not None and _StackExtruder is not None
    loops = _LoopExtruder(seq_padded, struct_padded, neighbor_bps=loop_base_size)
    stacks = _StackExtruder(seq_padded, struct_padded, stack_size=stack_size)

    # Clean format
    loops_cleaned = [_clean_feature(x) for x in loops]
    stacks_cleaned = [_clean_feature(x) for x in stacks]

    # Process hairpin loops
    if sep_base_stack:
        for loop in loops_cleaned.copy():
            seq_part, struct_part = loop.split("_")
            if hp_pattern.match(struct_part):
                # This is a hairpin loop
                # Extract loop without closing bp and base stack separately
                seq_unpadded = seq_part.replace("+", " ")
                hairpin_loop = _LoopExtruder(seq_unpadded, struct_part, neighbor_bps=0)[0]
                hairpin_stack = _StackExtruder(seq_unpadded, struct_part, stack_size=1)[0]

                # For tetraloops and shorter
                if len(seq_unpadded.replace(" ", "")) <= 6:
                    loops_cleaned.append(_clean_feature(hairpin_loop))
                else:
                    # Long hairpin loops get generic feature
                    loops_cleaned.append("NNNNN_.....")

                # Add hairpin mismatch parameter if requested
                if hairpin_mm:
                    # Format: first+last mismatch
                    seq_no_space = seq_unpadded.replace(" ", "")
                    loops_cleaned.append(
                        f"{seq_no_space[0]}{seq_no_space[1]}+{seq_no_space[-2]}{seq_no_space[-1]}_(.+.)"
                    )
                else:
                    # Add base stack (closing bp)
                    loops_cleaned.append(_clean_feature(hairpin_stack))

                # Remove original combined feature
                loops_cleaned.remove(loop)

            elif struct_part == "(..(+)..)":
                # Double mismatch (2x2 internal loop)
                seq_parts = seq_part.split("+")
                if len(seq_parts) == 2:
                    s1, s2 = seq_parts
                    # Mismatch bases
                    mm = f"{s1[1:3]}+{s2[1:3]}_..+.."
                    # Closing stack context
                    mm_stack = f"{s1[0]}+{s1[3]}+{s2[0]}+{s2[3]}_(+(+)+)"
                    loops_cleaned.append(mm)
                    loops_cleaned.append(mm_stack)
                    loops_cleaned.remove(loop)

            elif struct_part == "(...(+)...)":
                # Triple mismatch (3x3 internal loop) - reduced to double
                seq_parts = seq_part.split("+")
                if len(seq_parts) == 2:
                    s1, s2 = seq_parts
                    # Reduce to outer mismatch
                    mm = f"{s1[1]}{s1[3]}+{s2[1]}{s2[3]}_..+.."
                    mm_stack = f"{s1[0]}+{s1[4]}+{s2[0]}+{s2[4]}_(+(+)+)"
                    loops_cleaned.append(mm)
                    loops_cleaned.append(mm_stack)
                    loops_cleaned.remove(loop)

    # Combine features
    feature_list = loops_cleaned + stacks_cleaned

    # Remove any features without underscore (malformed)
    feature_list = [x for x in feature_list if "_" in x]

    return feature_list


def extract_features_standalone(seq: str, struct: str, stack_size: int = 2) -> list[str]:
    """
    Standalone feature extraction without RiboGraphViz.

    This is a simplified implementation for when RiboGraphViz is not available.
    It handles basic hairpin loops and NN stacks but may not capture all features.
    """
    features = []
    hp_pattern = re.compile(r"^\([.]+\)")

    pad = stack_size - 1
    pad_5p = "x"
    pad_3p = "y"

    # Pad sequence and structure
    seq_padded = pad_5p * pad + seq + pad_3p * pad
    struct_padded = "(" * pad + struct + ")" * pad

    n = len(struct)

    # Parse structure to find base pairs
    stack = []
    pairs = {}  # Maps opening to closing positions (in original struct)

    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j

    # Find hairpin loop (continuous dots)
    dot_regions = []
    start = None
    for i, c in enumerate(struct):
        if c == ".":
            if start is None:
                start = i
        else:
            if start is not None:
                dot_regions.append((start, i))
                start = None
    if start is not None:
        dot_regions.append((start, len(struct)))

    # Process each dot region
    for dot_start, dot_end in dot_regions:
        loop_size = dot_end - dot_start

        # Check if this is a hairpin loop (has closing bp)
        if dot_start > 0 and dot_end < n:
            if struct[dot_start - 1] == "(" and struct[dot_end] == ")":
                # Hairpin loop
                closing_5p = seq[dot_start - 1]
                closing_3p = seq[dot_end]
                loop_seq = seq[dot_start:dot_end]

                if loop_size <= 4:
                    # Triloop or tetraloop
                    features.append(f"{loop_seq}_{'.' * loop_size}")
                else:
                    # Longer loops
                    features.append("NNNNN_.....")

                # Base stack
                features.append(f"{closing_5p}+{closing_3p}_(+)")
            else:
                # Internal loop or bulge
                pass

    # Extract NN stacks from stem
    sorted_5p = sorted([k for k in pairs.keys() if struct[k] == "("])

    for i in range(len(sorted_5p) - 1):
        pos1 = sorted_5p[i]
        pos2 = sorted_5p[i + 1]

        # Check if consecutive
        if pos2 == pos1 + 1:
            j1 = pairs[pos1]  # Partner of pos1
            j2 = pairs[pos2]  # Partner of pos2

            # Ensure j2 = j1 - 1 (consecutive on 3' side)
            if j2 == j1 - 1:
                bp1_5 = seq[pos1]
                bp1_3 = seq[j1]
                bp2_5 = seq[pos2]
                bp2_3 = seq[j2]

                stack_feat = f"{bp1_5}{bp2_5}+{bp2_3}{bp1_3}_((+))"
                features.append(stack_feat)

    # Terminal stack with padding
    if sorted_5p:
        first_5p = sorted_5p[0]
        last_3p = pairs[first_5p]

        # 5' terminal
        term_5p = pad_5p if first_5p == 0 else seq[first_5p - 1]
        term_3p = pad_3p if last_3p == len(seq) - 1 else seq[last_3p + 1]

        term_feat = f"{term_5p}{seq[first_5p]}+{seq[last_3p]}{term_3p}_((+))"
        features.append(term_feat)

    return features


def extract_features(seq: str, struct: str, stack_size: int = 2, **kwargs) -> list[str]:
    """
    Extract features from sequence and structure.
    Uses RiboGraphViz if available, otherwise falls back to standalone.
    """
    if HAS_RIBOGRAPHVIZ:
        return extract_features_rgv(seq, struct, stack_size, **kwargs)
    else:
        return extract_features_standalone(seq, struct, stack_size)


@dataclass
class PredictionResult:
    """Result of thermodynamic prediction."""

    sequence: str
    structure: str
    dH: float
    dG_37: float
    Tm: float
    dS: float
    Tm_adjusted: Optional[float] = None
    gc_content: float = 0.0
    features: list[str] = field(default_factory=list)
    n_unknown_features: int = 0
    confidence: Optional[str] = None
    structure_type: Optional[str] = None
    prediction_method: str = "thermodynamic"
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sequence": self.sequence,
            "structure": self.structure,
            "dH": round(self.dH, 2),
            "dG_37": round(self.dG_37, 2),
            "Tm": round(self.Tm, 1) if not np.isnan(self.Tm) else None,
            "Tm_adjusted": round(self.Tm_adjusted, 1) if self.Tm_adjusted else None,
            "dS": round(self.dS, 4),
            "gc_content": round(self.gc_content, 1),
            "n_features": len(self.features),
            "n_unknown_features": self.n_unknown_features,
            "structure_type": self.structure_type,
            "prediction_method": self.prediction_method,
            "confidence": self.confidence,
            "warning": self.warning,
        }


@dataclass
class SaltConditions:
    """Salt concentration conditions for Tm adjustment."""

    Na: float = 0.05  # Sodium concentration in M (default 50 mM)
    Mg: float = 0.0  # Magnesium concentration in M (default 0, e.g., 0.002 for 2mM)
    K: float = 0.0  # Potassium concentration in M (treated as Na equivalent)

    @property
    def total_monovalent(self) -> float:
        """Total monovalent cation concentration (Na + K)."""
        return self.Na + self.K

    @property
    def has_magnesium(self) -> bool:
        """Check if Mg2+ is present."""
        return self.Mg > 0


class HairpinPredictor:
    """
    DNA hairpin Tm prediction engine.

    Uses dna24 Rich Parameter model (1,334 features) for accurate
    prediction of DNA hairpin folding thermodynamics.

    Based on high-throughput array experimental data (27,732 hairpins).
    Reference conditions: 1M Na+.

    Three prediction modes are available:
    1. GNN: Graph Neural Network (~1.8°C MAE, requires torch)
    2. Direct Tm regression: Linear model with structure-specific coefficients
    3. Thermodynamic: Tm = T_ref / (1 - dG/dH) - 273.15
    """

    T_REF = 273.15 + 37  # 310.15 K = 37°C

    def __init__(
        self,
        coefficients_file: Optional[Union[str, Path]] = None,
        reference_na: float = 1.0,
        use_direct_tm: bool = True,
        use_gnn: bool = False,
        auto_select_model: bool = False,
    ):
        """
        Initialize HairpinPredictor.

        Args:
            coefficients_file: Path to coefficients JSON file (default: dna24_enhanced.json)
            reference_na: Reference sodium concentration (default: 1.0M)
            use_direct_tm: Use direct Tm regression instead of thermodynamic calculation
            use_gnn: Use GNN model for all predictions (requires torch)
            auto_select_model: Automatically select best model per structure type:
                - triloop/tetraloop → Linear model (better for short loops)
                - other structures → GNN model (better overall)
                Overrides use_gnn when enabled. Requires torch for GNN.
        """
        self.reference_na = reference_na
        self.use_direct_tm = use_direct_tm
        self.use_gnn = use_gnn
        self.auto_select_model = auto_select_model

        self.coef_dH: dict[str, float] = {}
        self.coef_dG: dict[str, float] = {}
        self.coef_Tm: dict[str, float] = {}
        self.coef_Tm_by_type: dict[str, dict[str, float]] = {}
        self.metadata: dict = {}
        self._has_direct_tm: bool = False
        self._gnn_predictor = None
        self._gnn_available: bool = False

        # Initialize GNN if needed (use_gnn or auto_select_model)
        if use_gnn or auto_select_model:
            self._init_gnn(raise_on_error=use_gnn)

        if coefficients_file is None:
            enhanced_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "coefficients"
                / "dna24_enhanced.json"
            )
            standard_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "coefficients"
                / "dna24_coefficients.json"
            )
            if enhanced_path.exists():
                coefficients_file = enhanced_path
            elif standard_path.exists():
                coefficients_file = standard_path

        if coefficients_file:
            self.load_coefficients(coefficients_file)

    def _init_gnn(self, raise_on_error: bool = True) -> None:
        from ivt_hairpinstat.core.gnn_predictor import GNNPredictor, is_gnn_available

        if not is_gnn_available():
            self._gnn_available = False
            if raise_on_error:
                raise ImportError(
                    "GNN requires torch and torch_geometric. "
                    "Install with: pip install torch torch_geometric"
                )
            return

        self._gnn_predictor = GNNPredictor()
        self._gnn_available = True

    def load_coefficients(self, filepath: Union[str, Path]) -> None:
        """Load model coefficients from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self._has_direct_tm = False

        if "models" in data:
            self._load_enhanced_format(data)
        else:
            self._load_standard_format(data)

        print(f"Loaded {len(self.coef_dH)} dH coefficients")
        print(f"Loaded {len(self.coef_dG)} dG coefficients")
        if self._has_direct_tm:
            print(f"Loaded {len(self.coef_Tm)} direct Tm coefficients")
        if self.metadata:
            print(f"Model version: {self.metadata.get('version', 'unknown')}")

    def _load_enhanced_format(self, data: dict) -> None:
        """Load enhanced format with models dict containing global_dH, global_dG, global_Tm."""
        models = data["models"]

        if "global_dH" in models:
            self.coef_dH = models["global_dH"].get("coefficients", {})
        if "global_dG" in models:
            self.coef_dG = models["global_dG"].get("coefficients", {})
        if "global_Tm" in models:
            self.coef_Tm = models["global_Tm"].get("coefficients", {})
            self._has_direct_tm = len(self.coef_Tm) > 0

        self.coef_Tm_by_type: dict[str, dict[str, float]] = {}
        for model_name in ["tetraloop_Tm", "triloop_Tm", "watson_crick_Tm", "simple_Tm"]:
            if model_name in models:
                structure_type = model_name.replace("_Tm", "")
                self.coef_Tm_by_type[structure_type] = models[model_name].get("coefficients", {})

    def _load_standard_format(self, data: dict) -> None:
        """Load standard format with dH and dG at top level."""
        dH_data = data.get("dH", {})
        dG_data = data.get("dG", {})

        if isinstance(dH_data, dict) and "coefficients" in dH_data:
            self.coef_dH = dH_data["coefficients"]
        else:
            self.coef_dH = dH_data

        if isinstance(dG_data, dict) and "coefficients" in dG_data:
            self.coef_dG = dG_data["coefficients"]
        else:
            self.coef_dG = dG_data

    def predict(
        self,
        sequence: str,
        structure: Optional[str] = None,
        salt_conditions: Optional[SaltConditions] = None,
        DNA_conc: Optional[float] = None,
    ) -> PredictionResult:
        if structure is None:
            from ivt_hairpinstat.core.structure_parser import DNAStructureParser

            parser = DNAStructureParser()
            structure = parser.get_target_structure(sequence)

        structure_type = self._classify_structure(structure)

        use_gnn_for_this = self._should_use_gnn(structure_type)
        if use_gnn_for_this and self._gnn_predictor is not None:
            return self._predict_with_gnn(sequence, structure, salt_conditions)

        # Determine if duplex
        is_duplex = "+" in structure

        # Extract features using RiboGraphViz-compatible extraction
        features = extract_features(sequence, structure)

        n_unknown = sum(1 for f in features if f not in self.coef_dH and f not in self.coef_dG)

        dH = self._sum_coefficients(features, self.coef_dH)
        dG_37 = self._sum_coefficients(features, self.coef_dG)

        warning = None
        use_direct = self.use_direct_tm and self._has_direct_tm and not is_duplex
        prediction_method = "thermodynamic"

        if is_duplex and DNA_conc:
            is_self_comp = self._is_self_complementary(sequence)
            Tm = calculate_duplex_tm(dH, dG_37, DNA_conc, is_self_comp)
            prediction_method = "duplex"
        elif use_direct:
            Tm, prediction_method = self._predict_tm_direct(features, structure_type)
            n_unknown_tm = sum(1 for f in features if f not in self.coef_Tm)
            if n_unknown_tm > len(features) * 0.2:
                warning = f"Many features ({n_unknown_tm}/{len(features)}) not in direct Tm model"
        else:
            Tm, warning = self._calculate_tm_safe(dH, dG_37)

        # Calculate dS
        if not np.isnan(Tm) and Tm > -273.15:
            dS = dH / (Tm + 273.15)
        else:
            dS = 0.0

        # GC content
        seq_clean = sequence.replace("+", "") if isinstance(sequence, str) else "".join(sequence)
        gc = get_gc_content(seq_clean)

        # Salt adjustment (Na+ and/or Mg2+)
        Tm_adjusted = None
        if salt_conditions and not np.isnan(Tm):
            needs_adjustment = (
                salt_conditions.Na != self.reference_na or salt_conditions.has_magnesium
            )
            if needs_adjustment:
                n_bp = len([c for c in structure if c == "("])
                Tm_adjusted = get_salt_adjusted_tm(
                    Tm,
                    gc,
                    Na=salt_conditions.total_monovalent,
                    Mg=salt_conditions.Mg,
                    from_Na=self.reference_na,
                    n_bp=n_bp if n_bp > 1 else None,
                )

        # Estimate confidence based on structure type and feature coverage
        confidence = self._estimate_confidence(features, n_unknown, structure_type)

        return PredictionResult(
            sequence=sequence,
            structure=structure,
            dH=dH,
            dG_37=dG_37,
            Tm=Tm,
            Tm_adjusted=Tm_adjusted,
            dS=dS,
            gc_content=gc,
            features=features,
            n_unknown_features=n_unknown,
            confidence=confidence,
            structure_type=structure_type,
            prediction_method=prediction_method,
            warning=warning,
        )

    def _sum_coefficients(
        self,
        features: list[str],
        coefficients: dict[str, float],
    ) -> float:
        """Sum coefficients for all features."""
        total = 0.0
        for feature in features:
            if feature in coefficients:
                total += coefficients[feature]
        return total

    def _predict_tm_direct(
        self,
        features: list[str],
        structure_type: str,
    ) -> tuple[float, str]:
        """Predict Tm using direct regression, preferring structure-specific models."""
        type_coefs = getattr(self, "coef_Tm_by_type", {})

        if structure_type in type_coefs:
            Tm = self._sum_coefficients(features, type_coefs[structure_type])
            return Tm, f"direct_tm_{structure_type}"

        Tm = self._sum_coefficients(features, self.coef_Tm)
        return Tm, "direct_tm"

    def _predict_with_gnn(
        self,
        sequence: str,
        structure: str,
        salt_conditions: Optional[SaltConditions] = None,
    ) -> PredictionResult:
        assert self._gnn_predictor is not None, "GNN predictor not initialized"
        na_conc = salt_conditions.total_monovalent if salt_conditions else self.reference_na
        mg_conc = salt_conditions.Mg if salt_conditions else 0.0
        gnn_result = self._gnn_predictor.predict(sequence, structure, na_conc, mg_conc)

        structure_type = self._classify_structure(structure)
        seq_clean = sequence.replace("+", "") if isinstance(sequence, str) else "".join(sequence)
        gc = get_gc_content(seq_clean)

        return PredictionResult(
            sequence=sequence,
            structure=structure,
            dH=gnn_result.dH,
            dG_37=gnn_result.dG_37,
            Tm=gnn_result.Tm,
            Tm_adjusted=gnn_result.Tm_adjusted,
            dS=gnn_result.dS,
            gc_content=gc,
            features=[],
            n_unknown_features=0,
            confidence="high",
            structure_type=structure_type,
            prediction_method="gnn",
            warning=None,
        )

    def _calculate_tm_safe(self, dH: float, dG_37: float) -> tuple[float, Optional[str]]:
        """
        Calculate Tm with bounds checking.

        Returns:
            (Tm, warning_message)
        """
        warning = None

        # Check for problematic values
        if abs(dH) < 0.1:
            # dH too small, Tm calculation will be unstable
            return np.nan, "dH too small for reliable Tm calculation"

        ratio = dG_37 / dH

        if ratio >= 1.0:
            # dG/dH >= 1 means Tm would be infinite or negative
            return np.nan, "dG/dH ratio indicates structure is unstable at all temperatures"

        if ratio < 0:
            # Negative ratio can give very high Tm
            warning = "Unusual dG/dH ratio, prediction may be unreliable"

        # Calculate Tm: Tm = T_ref / (1 - dG/dH) - 273.15
        Tm = self.T_REF / (1 - ratio) - 273.15

        # Bounds check
        if Tm < -50:
            warning = f"Predicted Tm ({Tm:.1f}°C) is unusually low"
        elif Tm > 150:
            warning = f"Predicted Tm ({Tm:.1f}°C) is unusually high"

        return Tm, warning

    def _classify_structure(self, structure: str, series: str = "") -> str:
        """
        Classify structure type for model selection.

        Returns one of: tetraloop, triloop, watson_crick, simple_hairpin,
                       bulge, mismatch, duplex, complex
        """
        series_lower = series.lower() if series else ""

        if "tetra" in series_lower or "tloop" in series_lower:
            return "tetraloop"
        if "tri" in series_lower:
            return "triloop"
        if "watson" in series_lower or "wc" in series_lower:
            return "watson_crick"
        if "bulge" in series_lower:
            return "bulge"
        if "mismatch" in series_lower or "mm" in series_lower:
            return "mismatch"

        if "+" in structure:
            return "duplex"

        dot_matches = re.findall(r"\.+", structure)
        if len(dot_matches) == 0:
            return "watson_crick"
        elif len(dot_matches) == 1:
            loop_size = len(dot_matches[0])
            if loop_size == 3:
                return "triloop"
            elif loop_size == 4:
                return "tetraloop"
            elif loop_size <= 6:
                return "simple_hairpin"
            else:
                return "complex"
        else:
            return "complex"

    def _should_use_gnn(self, structure_type: str) -> bool:
        if self.use_gnn:
            return True
        if not self.auto_select_model:
            return False
        if not self._gnn_available:
            return False
        linear_preferred = {"triloop", "tetraloop"}
        return structure_type not in linear_preferred

    def _is_self_complementary(self, sequence: str | list[str]) -> bool:
        if isinstance(sequence, list):
            return sequence[0] == sequence[1]
        elif "+" in sequence:
            parts = sequence.split("+")
            return parts[0] == parts[1] if len(parts) == 2 else False
        return True

    def _estimate_confidence(
        self,
        features: list[str],
        n_unknown: int,
        structure_type: str,
    ) -> str:
        """
        Estimate prediction confidence based on feature coverage and structure type.

        Returns:
            "high" - simple structure, all features known
            "medium" - some unknown features or moderately complex structure
            "low" - many unknown features or complex structure
        """
        if not features:
            return "low"

        coverage = 1 - (n_unknown / len(features))

        # Structure type affects confidence
        simple_structures = {"triloop", "tetraloop", "hairpin_simple", "duplex"}

        if structure_type in simple_structures and coverage >= 0.95:
            return "high"
        elif structure_type in simple_structures and coverage >= 0.8:
            return "medium"
        elif coverage >= 0.9:
            return "medium"
        else:
            return "low"

    def predict_batch(
        self,
        sequences: list[str],
        structures: Optional[list[str]] = None,
        salt_conditions: Optional[SaltConditions] = None,
        batch_size: int = 64,
    ) -> list[PredictionResult]:
        struct_list: list[str] = []
        if structures:
            struct_list = list(structures)
        else:
            from ivt_hairpinstat.core.structure_parser import DNAStructureParser

            parser = DNAStructureParser()
            struct_list = [parser.get_target_structure(seq) for seq in sequences]

        if self.use_gnn and self._gnn_predictor is not None:
            na_conc = salt_conditions.total_monovalent if salt_conditions else self.reference_na
            mg_conc = salt_conditions.Mg if salt_conditions else 0.0
            gnn_results = self._gnn_predictor.model.predict_batch(
                sequences, struct_list, na_conc, mg_conc, batch_size
            )
            results = []
            for gnn_r in gnn_results:
                structure_type = self._classify_structure(gnn_r.structure)
                results.append(
                    PredictionResult(
                        sequence=gnn_r.sequence,
                        structure=gnn_r.structure,
                        dH=gnn_r.dH,
                        dG_37=gnn_r.dG_37,
                        Tm=gnn_r.Tm,
                        Tm_adjusted=gnn_r.Tm_adjusted,
                        dS=gnn_r.dS,
                        gc_content=gnn_r.gc_content,
                        features=[],
                        n_unknown_features=0,
                        confidence="high",
                        structure_type=structure_type,
                        prediction_method="gnn",
                    )
                )
            return results

        if self.auto_select_model and self._gnn_available:
            return self._predict_batch_auto(sequences, struct_list, salt_conditions, batch_size)

        results = []
        for seq, struct in zip(sequences, struct_list):
            result = self.predict(seq, struct, salt_conditions)
            results.append(result)
        return results

    def _predict_batch_auto(
        self,
        sequences: list[str],
        structures: list[str],
        salt_conditions: Optional[SaltConditions],
        batch_size: int,
    ) -> list[PredictionResult]:
        linear_indices = []
        gnn_indices = []
        structure_types = []

        for i, struct in enumerate(structures):
            stype = self._classify_structure(struct)
            structure_types.append(stype)
            if stype in {"triloop", "tetraloop"}:
                linear_indices.append(i)
            else:
                gnn_indices.append(i)

        results: list[Optional[PredictionResult]] = [None] * len(sequences)

        for i in linear_indices:
            result = self.predict(sequences[i], structures[i], salt_conditions)
            results[i] = result

        if gnn_indices and self._gnn_predictor is not None:
            gnn_seqs = [sequences[i] for i in gnn_indices]
            gnn_structs = [structures[i] for i in gnn_indices]
            na_conc = salt_conditions.total_monovalent if salt_conditions else self.reference_na
            mg_conc = salt_conditions.Mg if salt_conditions else 0.0

            gnn_results = self._gnn_predictor.model.predict_batch(
                gnn_seqs, gnn_structs, na_conc, mg_conc, batch_size
            )

            for idx, gnn_r in zip(gnn_indices, gnn_results):
                results[idx] = PredictionResult(
                    sequence=gnn_r.sequence,
                    structure=gnn_r.structure,
                    dH=gnn_r.dH,
                    dG_37=gnn_r.dG_37,
                    Tm=gnn_r.Tm,
                    Tm_adjusted=gnn_r.Tm_adjusted,
                    dS=gnn_r.dS,
                    gc_content=gnn_r.gc_content,
                    features=[],
                    n_unknown_features=0,
                    confidence="high",
                    structure_type=structure_types[idx],
                    prediction_method="gnn",
                )

        return [r for r in results if r is not None]

    def get_feature_contributions(
        self,
        sequence: str,
        structure: str,
    ) -> dict:
        """
        Get detailed breakdown of feature contributions to dH and dG.

        Returns:
            Dictionary with feature breakdown
        """
        features = extract_features(sequence, structure)

        contributions = []
        for f in features:
            dH_contrib = self.coef_dH.get(f, 0.0)
            dG_contrib = self.coef_dG.get(f, 0.0)
            contributions.append(
                {
                    "feature": f,
                    "dH": round(dH_contrib, 3),
                    "dG": round(dG_contrib, 3),
                    "known": f in self.coef_dH or f in self.coef_dG,
                }
            )

        total_dH = sum(c["dH"] for c in contributions)
        total_dG = sum(c["dG"] for c in contributions)

        return {
            "sequence": sequence,
            "structure": structure,
            "features": contributions,
            "total_dH": round(total_dH, 2),
            "total_dG": round(total_dG, 2),
            "n_features": len(contributions),
            "n_unknown": sum(1 for c in contributions if not c["known"]),
        }

    def get_model_info(self) -> dict:
        return {
            "version": self.metadata.get("version", "unknown"),
            "description": self.metadata.get("description", ""),
            "n_dH_coefficients": len(self.coef_dH),
            "n_dG_coefficients": len(self.coef_dG),
            "n_Tm_coefficients": len(self.coef_Tm),
            "has_direct_tm": self._has_direct_tm,
            "use_direct_tm": self.use_direct_tm,
            "reference_na": self.reference_na,
            "gnn_available": self._gnn_available,
        }

    def predict_ensemble(
        self,
        sequence: str,
        structure: Optional[str] = None,
        salt_conditions: Optional[SaltConditions] = None,
        weights: Optional[tuple[float, float]] = None,
    ) -> PredictionResult:
        if structure is None:
            from ivt_hairpinstat.core.structure_parser import DNAStructureParser

            parser = DNAStructureParser()
            structure = parser.get_target_structure(sequence)

        if not self._gnn_available:
            return self.predict(sequence, structure, salt_conditions)

        if self._gnn_predictor is None:
            self._init_gnn(raise_on_error=True)

        linear_result = self._predict_linear(sequence, structure, salt_conditions)

        na_conc = salt_conditions.total_monovalent if salt_conditions else self.reference_na
        mg_conc = salt_conditions.Mg if salt_conditions else 0.0
        assert self._gnn_predictor is not None
        gnn_raw = self._gnn_predictor.predict(sequence, structure, na_conc, mg_conc)

        structure_type = self._classify_structure(structure)

        if weights is None:
            if structure_type in {"triloop", "tetraloop"}:
                w_linear, w_gnn = 0.7, 0.3
            else:
                w_linear, w_gnn = 0.3, 0.7
        else:
            w_linear, w_gnn = weights

        Tm_ensemble = w_linear * linear_result.Tm + w_gnn * gnn_raw.Tm
        dH_ensemble = w_linear * linear_result.dH + w_gnn * gnn_raw.dH
        dG_ensemble = w_linear * linear_result.dG_37 + w_gnn * gnn_raw.dG_37

        T_ref = 273.15 + 37
        if abs(dH_ensemble) > 0.1 and Tm_ensemble > -200:
            dS_ensemble = dH_ensemble / (Tm_ensemble + 273.15)
        else:
            dS_ensemble = 0.0

        Tm_adjusted = None
        if salt_conditions:
            needs_adjustment = (
                salt_conditions.Na != self.reference_na or salt_conditions.has_magnesium
            )
            if needs_adjustment:
                gc = linear_result.gc_content
                n_bp = len([c for c in structure if c == "("])
                Tm_adjusted = get_salt_adjusted_tm(
                    Tm_ensemble,
                    gc,
                    Na=salt_conditions.total_monovalent,
                    Mg=salt_conditions.Mg,
                    from_Na=self.reference_na,
                    n_bp=n_bp if n_bp > 1 else None,
                )

        return PredictionResult(
            sequence=sequence,
            structure=structure,
            dH=dH_ensemble,
            dG_37=dG_ensemble,
            Tm=Tm_ensemble,
            Tm_adjusted=Tm_adjusted,
            dS=dS_ensemble,
            gc_content=linear_result.gc_content,
            features=linear_result.features,
            n_unknown_features=linear_result.n_unknown_features,
            confidence="high",
            structure_type=structure_type,
            prediction_method=f"ensemble_{w_linear:.1f}L_{w_gnn:.1f}G",
            warning=None,
        )

    def _predict_linear(
        self,
        sequence: str,
        structure: str,
        salt_conditions: Optional[SaltConditions] = None,
    ) -> PredictionResult:
        original_use_gnn = self.use_gnn
        original_auto = self.auto_select_model
        self.use_gnn = False
        self.auto_select_model = False
        try:
            result = self.predict(sequence, structure, salt_conditions)
        finally:
            self.use_gnn = original_use_gnn
            self.auto_select_model = original_auto
        return result


# Backwards compatibility alias
ThermoPredictor = HairpinPredictor
