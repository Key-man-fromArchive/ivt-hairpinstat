"""
Feature extractor for DNA sequences.

Extracts structural features from DNA sequences for use with
the Rich Parameter linear regression model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ivt_hairpinstat.core.structure_parser import (
    DNAStructureParser,
    ParsedStructure,
    MotifType,
)


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""

    stack_size: int = 2  # NN (2) or NNN (3)
    sep_base_stack: bool = True  # Separate hairpin base stack
    symmetry: bool = False  # Treat symmetric motifs as same
    fit_intercept: bool = False  # Add intercept term
    pad_char_5p: str = "x"  # Padding character for 5' end
    pad_char_3p: str = "y"  # Padding character for 3' end


class FeatureExtractor:
    """
    Extracts features from DNA sequences for thermodynamic prediction.

    Features are extracted based on the structural decomposition of
    DNA hairpins and duplexes into stacks, loops, mismatches, and bulges.

    Compatible with the dna24 Rich Parameter model format.
    """

    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        self.config = config or FeatureExtractionConfig()
        self.parser = DNAStructureParser()

    def extract_features(
        self,
        sequence: str,
        structure: str,
    ) -> list[str]:
        """
        Extract feature list from a DNA sequence and structure.

        Args:
            sequence: DNA sequence (ATCG)
            structure: Dot-bracket notation

        Returns:
            List of feature strings in model format
        """
        features = []

        # Handle duplex vs hairpin
        if "+" in structure:
            features = self._extract_duplex_features(sequence, structure)
        else:
            features = self._extract_hairpin_features(sequence, structure)

        # Add intercept if configured
        if self.config.fit_intercept:
            features.append("intercept")

        return features

    def _extract_hairpin_features(
        self,
        sequence: str,
        structure: str,
    ) -> list[str]:
        """Extract features from a hairpin structure."""
        features = []
        cfg = self.config

        # Pad sequence and structure
        pad = cfg.stack_size - 1
        seq_padded = cfg.pad_char_5p * pad + sequence + cfg.pad_char_3p * pad
        struct_padded = "(" * pad + structure + ")" * pad

        # Find loop region
        loop_start = structure.find(".")
        loop_end = structure.rfind(".") + 1

        if loop_start == -1:
            # No loop - unstructured
            return features

        loop_size = loop_end - loop_start
        loop_seq = sequence[loop_start:loop_end]

        # Extract loop features
        if cfg.sep_base_stack:
            # Separate loop body from closing base pair
            if loop_size <= 4:
                # Triloop or tetraloop - use full sequence
                loop_feature = f"{loop_seq}_{'.' * loop_size}"
                features.append(loop_feature)
            else:
                # Longer loops - use generic feature
                features.append("NNNNN_.....")

            # Add hairpin mismatch (closing bp context)
            if loop_start > 0 and loop_end < len(sequence):
                closing_5p = sequence[loop_start - 1]
                closing_3p = sequence[loop_end]
                first_loop = sequence[loop_start] if loop_start < len(sequence) else "N"
                last_loop = sequence[loop_end - 1] if loop_end > 0 else "N"
                mm_feature = f"{closing_5p}{first_loop}+{last_loop}{closing_3p}_(.+.)"
                features.append(mm_feature)
        else:
            # Include closing bp in loop feature
            full_loop = sequence[loop_start - 1 : loop_end + 1]
            features.append(f"{full_loop}_({loop_seq})")

        # Extract stack features (NN or NNN)
        features.extend(
            self._extract_stem_stacks(seq_padded, struct_padded, loop_start + pad, loop_end + pad)
        )

        return features

    def _extract_stem_stacks(
        self,
        seq_padded: str,
        struct_padded: str,
        loop_start: int,
        loop_end: int,
    ) -> list[str]:
        """Extract nearest-neighbor stacks from stem region."""
        features = []
        cfg = self.config
        n = len(seq_padded)

        # 5' side positions
        i = 0
        # 3' side positions (from end)
        j = n - 1

        while i < loop_start - cfg.stack_size + 1:
            # Extract NN or NNN stack
            stack_5p = seq_padded[i : i + cfg.stack_size]
            stack_3p = seq_padded[j - cfg.stack_size + 1 : j + 1][::-1]

            # Format: "AB+CD_((+))" for NN stack
            stack_struct = "(" * cfg.stack_size + "+" + ")" * cfg.stack_size
            stack_feature = f"{stack_5p}+{stack_3p}_{stack_struct}"

            if cfg.symmetry:
                stack_feature = self._symmetrize_stack(stack_feature)

            features.append(stack_feature)

            i += 1
            j -= 1

        return features

    def _extract_duplex_features(
        self,
        sequence: str,
        structure: str,
    ) -> list[str]:
        """Extract features from a duplex structure."""
        features = []
        cfg = self.config

        # Parse duplex sequence
        if isinstance(sequence, list):
            seq1, seq2 = sequence
        else:
            # Assume sequence is "seq1+seq2" or needs to be split
            if "+" in sequence:
                seq1, seq2 = sequence.split("+")
            else:
                # Self-complementary duplex
                half = len(sequence) // 2
                seq1 = sequence[:half]
                seq2 = self._reverse_complement(sequence[half:])

        # Pad sequences
        pad = cfg.stack_size - 1
        seq1_padded = cfg.pad_char_5p * pad + seq1 + cfg.pad_char_3p * pad
        seq2_padded = cfg.pad_char_5p * pad + seq2 + cfg.pad_char_3p * pad

        n = min(len(seq1), len(seq2))

        # Extract stacks along the duplex
        for i in range(n - cfg.stack_size + 1):
            stack_5p = seq1_padded[i : i + cfg.stack_size]
            # 3' strand runs antiparallel
            j = len(seq2) - i - 1
            stack_3p = seq2_padded[j - cfg.stack_size + 2 : j + 2][::-1]

            stack_struct = "(" * cfg.stack_size + "+" + ")" * cfg.stack_size
            stack_feature = f"{stack_5p}+{stack_3p}_{stack_struct}"

            # Check for mismatches
            bp1 = seq1[i] + seq2[-(i + 1)] if i < len(seq1) and i < len(seq2) else "NN"
            if bp1 not in {"AT", "TA", "CG", "GC"}:
                # This is a mismatch stack
                pass  # Keep the feature, model will handle it

            if cfg.symmetry:
                stack_feature = self._symmetrize_stack(stack_feature)

            features.append(stack_feature)

        return features

    def _symmetrize_stack(self, stack_feature: str) -> str:
        """Sort stack to treat symmetric motifs as equivalent."""
        seq_part, struct_part = stack_feature.split("_")
        parts = seq_part.split("+")
        sorted_parts = "+".join(sorted(parts))
        return f"{sorted_parts}_{struct_part}"

    def _reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
        return "".join(complement.get(base, "N") for base in reversed(sequence.upper()))

    def get_feature_counts(
        self,
        sequence: str,
        structure: str,
    ) -> dict[str, int]:
        """
        Get feature counts as a dictionary.

        Args:
            sequence: DNA sequence
            structure: Dot-bracket structure

        Returns:
            Dictionary mapping feature names to counts
        """
        features = self.extract_features(sequence, structure)
        return dict(Counter(features))

    def extract_mismatch_features(
        self,
        sequence: str,
        structure: str,
    ) -> list[dict]:
        """
        Extract detailed mismatch information for reporting.

        Returns:
            List of mismatch info dicts with position, type, and context
        """
        mismatches = []
        parsed = self.parser.parse(sequence, structure)

        for motif in parsed.motifs:
            if motif.motif_type == MotifType.INTERNAL_LOOP:
                mismatch_info = {
                    "position": motif.position,
                    "sequence": motif.sequence,
                    "type": self._classify_mismatch(motif.sequence),
                    "feature": motif.to_feature_string(),
                }
                mismatches.append(mismatch_info)

        return mismatches

    def _classify_mismatch(self, mismatch_seq: str) -> str:
        """Classify mismatch type."""
        parts = mismatch_seq.replace(" ", "+").split("+")
        if len(parts) == 2:
            bp = parts[0] + parts[1]
            if bp in {"GT", "TG"}:
                return "G-T wobble"
            elif bp in {"AC", "CA"}:
                return "A-C mismatch"
            elif bp in {"AA"}:
                return "A-A mismatch"
            elif bp in {"CC"}:
                return "C-C mismatch"
            elif bp in {"GG"}:
                return "G-G mismatch"
            elif bp in {"TT"}:
                return "T-T mismatch"
            else:
                return f"{parts[0]}-{parts[1]} mismatch"
        return "unknown"
