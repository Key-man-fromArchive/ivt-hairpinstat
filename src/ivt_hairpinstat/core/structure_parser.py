"""
DNA structure parser for dot-bracket notation.

Parses hairpin and duplex structures to identify stems, loops,
mismatches, and bulges without external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class StructureType(Enum):
    """Types of DNA secondary structures."""

    HAIRPIN = "hairpin"
    DUPLEX = "duplex"
    UNSTRUCTURED = "unstructured"


class MotifType(Enum):
    """Types of structural motifs."""

    STACK = "stack"
    HAIRPIN_LOOP = "hairpin_loop"
    INTERNAL_LOOP = "internal_loop"  # includes mismatches
    BULGE = "bulge"
    TERMINAL = "terminal"


@dataclass
class StructuralMotif:
    """Represents a structural motif in a DNA structure."""

    motif_type: MotifType
    sequence: str
    structure: str
    position: tuple[int, int]  # start, end positions in original sequence

    def to_feature_string(self) -> str:
        """Convert motif to feature string format used in model."""
        seq_formatted = self.sequence.replace(" ", "+")
        struct_formatted = self.structure.replace(" ", "+")
        return f"{seq_formatted}_{struct_formatted}"


@dataclass
class ParsedStructure:
    """Result of parsing a DNA structure."""

    sequence: str
    structure: str
    structure_type: StructureType
    motifs: list[StructuralMotif] = field(default_factory=list)

    @property
    def stem_length(self) -> int:
        """Calculate total stem length (paired bases)."""
        return self.structure.count("(")

    @property
    def loop_length(self) -> int:
        """Calculate loop length for hairpins."""
        if self.structure_type == StructureType.HAIRPIN:
            return self.structure.count(".")
        return 0


class DNAStructureParser:
    """
    Parser for DNA secondary structures in dot-bracket notation.

    Supports:
    - Hairpin structures: (((...)))
    - Duplex structures: (((+)))
    - Mismatches within stems
    - Bulges
    - Various loop sizes
    """

    COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
    WC_PAIRS = {"AT", "TA", "CG", "GC"}

    def __init__(self):
        self._hairpin_pattern = re.compile(r"^\(*\.+\)*$")

    def parse(self, sequence: str, structure: str) -> ParsedStructure:
        """
        Parse a DNA sequence with its dot-bracket structure.

        Args:
            sequence: DNA sequence (ATCG) or list for duplex ['seq1', 'seq2']
            structure: Dot-bracket notation

        Returns:
            ParsedStructure with identified motifs
        """
        # Handle duplex vs hairpin
        if "+" in structure or isinstance(sequence, list):
            return self._parse_duplex(sequence, structure)
        else:
            return self._parse_hairpin(sequence, structure)

    def _parse_hairpin(self, sequence: str, structure: str) -> ParsedStructure:
        """Parse a hairpin structure."""
        result = ParsedStructure(
            sequence=sequence, structure=structure, structure_type=StructureType.HAIRPIN, motifs=[]
        )

        # Find loop region
        loop_start = structure.find(".")
        loop_end = structure.rfind(".") + 1

        if loop_start == -1:
            # No loop found - might be unstructured
            result.structure_type = StructureType.UNSTRUCTURED
            return result

        loop_seq = sequence[loop_start:loop_end]
        loop_struct = structure[loop_start:loop_end]

        # Add hairpin loop motif
        result.motifs.append(
            StructuralMotif(
                motif_type=MotifType.HAIRPIN_LOOP,
                sequence=loop_seq,
                structure=loop_struct,
                position=(loop_start, loop_end),
            )
        )

        # Parse stem for stacks, mismatches, bulges
        self._parse_stem(sequence, structure, loop_start, loop_end, result)

        return result

    def _parse_duplex(self, sequence: str | list[str], structure: str) -> ParsedStructure:
        """Parse a duplex structure."""
        # Handle sequence format
        if isinstance(sequence, list):
            seq1, seq2 = sequence
            seq_combined = f"{seq1}+{seq2}"
        else:
            seq_combined = sequence
            parts = structure.split("+")
            if len(parts) == 2:
                mid = len(parts[0])
                seq1 = sequence[:mid]
                seq2 = sequence[mid:]
            else:
                seq1 = sequence
                seq2 = self._reverse_complement(sequence)

        result = ParsedStructure(
            sequence=seq_combined,
            structure=structure,
            structure_type=StructureType.DUPLEX,
            motifs=[],
        )

        # Parse duplex stem
        self._parse_duplex_stem(seq1, seq2, structure, result)

        return result

    def _parse_stem(
        self, sequence: str, structure: str, loop_start: int, loop_end: int, result: ParsedStructure
    ) -> None:
        """Parse stem region for stacks and other motifs."""
        n = len(sequence)

        # 5' side of stem (before loop)
        i = 0
        # 3' side of stem (after loop)
        j = n - 1

        while i < loop_start and j >= loop_end:
            # Check for Watson-Crick pair
            bp = sequence[i] + sequence[j]

            if structure[i] == "(" and structure[j] == ")":
                # This is a paired position
                if bp in self.WC_PAIRS:
                    # Watson-Crick stack
                    # Look for next-nearest neighbor context
                    if i + 1 < loop_start and j - 1 >= loop_end:
                        next_bp = sequence[i + 1] + sequence[j - 1]
                        stack_seq = f"{sequence[i]}{sequence[i + 1]}+{sequence[j - 1]}{sequence[j]}"
                        result.motifs.append(
                            StructuralMotif(
                                motif_type=MotifType.STACK,
                                sequence=stack_seq,
                                structure="((+))",
                                position=(i, j),
                            )
                        )
                else:
                    # Mismatch (non-WC pair in stem)
                    result.motifs.append(
                        StructuralMotif(
                            motif_type=MotifType.INTERNAL_LOOP,
                            sequence=f"{sequence[i]}+{sequence[j]}",
                            structure="(.+.)",
                            position=(i, j),
                        )
                    )
            elif structure[i] == "." and structure[j] == ")":
                # 5' bulge
                bulge_end = i
                while bulge_end < loop_start and structure[bulge_end] == ".":
                    bulge_end += 1
                bulge_seq = sequence[i:bulge_end]
                result.motifs.append(
                    StructuralMotif(
                        motif_type=MotifType.BULGE,
                        sequence=bulge_seq,
                        structure="." * len(bulge_seq),
                        position=(i, bulge_end),
                    )
                )
                i = bulge_end - 1
            elif structure[i] == "(" and structure[j] == ".":
                # 3' bulge
                bulge_start = j
                while bulge_start >= loop_end and structure[bulge_start] == ".":
                    bulge_start -= 1
                bulge_seq = sequence[bulge_start + 1 : j + 1]
                result.motifs.append(
                    StructuralMotif(
                        motif_type=MotifType.BULGE,
                        sequence=bulge_seq,
                        structure="." * len(bulge_seq),
                        position=(bulge_start + 1, j + 1),
                    )
                )
                j = bulge_start + 1

            i += 1
            j -= 1

        # Add terminal motif
        if len(sequence) > 0:
            terminal_bp = sequence[0] + sequence[-1]
            result.motifs.append(
                StructuralMotif(
                    motif_type=MotifType.TERMINAL,
                    sequence=terminal_bp,
                    structure="(+)",
                    position=(0, n - 1),
                )
            )

    def _parse_duplex_stem(
        self, seq1: str, seq2: str, structure: str, result: ParsedStructure
    ) -> None:
        """Parse duplex stem for stacks and mismatches."""
        n = min(len(seq1), len(seq2))

        for i in range(n - 1):
            # Stack: consecutive base pairs
            bp1 = seq1[i] + seq2[-(i + 1)]
            bp2 = seq1[i + 1] + seq2[-(i + 2)] if i + 1 < n else None

            if bp2:
                stack_seq = f"{seq1[i]}{seq1[i + 1]}+{seq2[-(i + 2)]}{seq2[-(i + 1)]}"

                # Check if it's a mismatch
                if bp1 not in self.WC_PAIRS or bp2 not in self.WC_PAIRS:
                    result.motifs.append(
                        StructuralMotif(
                            motif_type=MotifType.INTERNAL_LOOP,
                            sequence=stack_seq,
                            structure="((+))",
                            position=(i, i + 1),
                        )
                    )
                else:
                    result.motifs.append(
                        StructuralMotif(
                            motif_type=MotifType.STACK,
                            sequence=stack_seq,
                            structure="((+))",
                            position=(i, i + 1),
                        )
                    )

    def _reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of a DNA sequence."""
        return "".join(self.COMPLEMENT.get(base, "N") for base in reversed(sequence.upper()))

    def get_target_structure(self, sequence: str, loop_size: int = 4) -> str:
        """
        Generate target hairpin structure for a sequence.

        Assumes a simple hairpin with specified loop size.

        Args:
            sequence: DNA sequence
            loop_size: Size of the hairpin loop (default 4 for tetraloop)

        Returns:
            Dot-bracket structure string
        """
        n = len(sequence)
        stem_len = (n - loop_size) // 2
        return "(" * stem_len + "." * loop_size + ")" * stem_len

    def identify_mismatches(self, sequence: str, structure: str) -> list[tuple[int, int, str]]:
        """
        Identify all mismatch positions in a structure.

        Returns:
            List of (position_5prime, position_3prime, mismatch_type) tuples
        """
        mismatches = []
        parsed = self.parse(sequence, structure)

        for motif in parsed.motifs:
            if motif.motif_type == MotifType.INTERNAL_LOOP:
                seq_parts = motif.sequence.split("+")
                if len(seq_parts) == 2:
                    mm_type = f"{seq_parts[0]}_{seq_parts[1]}"
                    mismatches.append((motif.position[0], motif.position[1], mm_type))

        return mismatches
