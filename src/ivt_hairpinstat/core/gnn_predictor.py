"""
GNN-based Hairpin Tm predictor.

Uses pretrained Graph Neural Network from dna24 paper (Nature Communications 2025).
Achieves ~1.8°C MAE on test set, approaching experimental measurement error (1.28°C).

Architecture:
- 4x Graph Transformer layers (125 hidden channels)
- Set2Set pooling (10 processing steps)
- FC layer (128 channels) with dropout
- Total: 287,136 parameters

Requires: torch, torch_geometric
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np

from ivt_hairpinstat.core.thermodynamics import get_salt_adjusted_tm

# Lazy imports for torch (optional dependency)
_torch = None
_torch_geometric = None


def _import_torch():
    """Lazy import torch and torch_geometric."""
    global _torch, _torch_geometric
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GNN predictor. Install with: pip install torch"
            )
    if _torch_geometric is None:
        try:
            import torch_geometric

            _torch_geometric = torch_geometric
        except ImportError:
            raise ImportError(
                "PyTorch Geometric is required for GNN predictor. "
                "Install with: pip install torch_geometric"
            )
    return _torch, _torch_geometric


@dataclass
class GNNPredictionResult:
    """Result from GNN prediction."""

    sequence: str
    structure: str
    dH: float
    Tm: float
    dG_37: float
    dS: float
    Tm_adjusted: Optional[float] = None
    gc_content: float = 0.0
    prediction_method: str = "gnn"


class GNNModel:
    """
    Graph Neural Network for DNA hairpin Tm prediction.

    Minimal implementation without wandb dependency.
    """

    # Default model config (from paper)
    DEFAULT_CONFIG = {
        "hidden_channels": 125,
        "pooling": "Set2Set",
        "processing_steps": 10,
        "n_graphconv_layer": 4,
        "n_linear_layer": 1,
        "linear_hidden_channels": [128],
        "graphconv_dropout": 0.012732466797412492,
        "linear_dropout": 0.49,
        "concat": False,
    }

    # Normalization statistics from training set
    DEFAULT_SUMSTATS = {
        "dH_min": -56.0,
        "dH_max": -5.0,
        "Tm_min": 21.0,
        "Tm_max": 82.0,
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[dict] = None,
        sumstats: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize GNN model.

        Args:
            model_path: Path to pretrained model weights (.pt file)
            config: Model configuration (uses default if not provided)
            sumstats: Normalization statistics (uses default if not provided)
            device: 'cpu' or 'cuda' (auto-detected if not provided)
        """
        torch, _ = _import_torch()

        self.config = config or self.DEFAULT_CONFIG.copy()
        self.sumstats = sumstats or self.DEFAULT_SUMSTATS.copy()

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Build model
        self.model = self._build_model()

        # Load weights if provided
        if model_path is None:
            # Try default location
            default_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "models"
                / "gnn_state_dict_ancient-sound-259.pt"
            )
            if default_path.exists():
                model_path = default_path

        if model_path:
            self.load_weights(model_path)

    def _build_model(self):
        """Build the GNN architecture."""
        torch, _ = _import_torch()
        from torch.nn import Linear, ModuleList
        import torch.nn.functional as F
        from torch_geometric.nn import TransformerConv
        from torch_geometric.nn.aggr import Set2Set

        config = self.config

        class _GNN(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                torch.manual_seed(12345)

                num_node_features = 4  # A, T, C, G
                num_edge_features = 3  # 5'->3', 3'->5', H-bond
                num_heads = 1

                self.graphconv_dropout = cfg["graphconv_dropout"]
                self.linear_dropout = cfg["linear_dropout"]
                self.concat = cfg["concat"]

                # Graph transformer layers
                conv_list = [
                    TransformerConv(
                        num_node_features,
                        cfg["hidden_channels"],
                        heads=num_heads,
                        edge_dim=num_edge_features,
                        dropout=self.graphconv_dropout,
                    )
                ]
                for _ in range(cfg["n_graphconv_layer"] - 1):
                    conv_list.append(
                        TransformerConv(
                            cfg["hidden_channels"],
                            cfg["hidden_channels"],
                            heads=num_heads,
                            edge_dim=num_edge_features,
                            dropout=self.graphconv_dropout,
                        )
                    )
                self.convs = ModuleList(conv_list)

                # Pooling
                n_pool = cfg["hidden_channels"]
                self.aggr = Set2Set(n_pool, processing_steps=cfg["processing_steps"])

                # Linear layers
                linear_list = [Linear(2 * n_pool, cfg["linear_hidden_channels"][0])]
                linear_list.append(Linear(cfg["linear_hidden_channels"][-1], 2))  # dH, Tm
                self.linears = ModuleList(linear_list)

            def forward(self, x, edge_index, edge_attr, batch):
                # Graph convolutions
                for conv in self.convs:
                    x = conv(x, edge_index, edge_attr)
                    x = F.leaky_relu(x)
                    x = F.dropout(x, p=self.graphconv_dropout, training=self.training)

                # Pooling
                x = self.aggr(x, batch)

                # Linear layers
                for i, lin in enumerate(self.linears):
                    x = lin(x)
                    if i < len(self.linears) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.linear_dropout, training=self.training)

                return torch.flatten(x)

        model = _GNN(config).to(self.device)
        return model

    def load_weights(self, model_path: Union[str, Path]) -> None:
        """Load pretrained weights."""
        torch, _ = _import_torch()

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"GNN model loaded: {n_params:,} parameters, device={self.device}")

    def _prepare_input(self, sequence: str, structure: str):
        """Convert sequence and structure to graph tensors."""
        torch, _ = _import_torch()

        edge_list, edge_feat = self._dotbracket_to_edges(structure)
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        edge_attr = torch.tensor(edge_feat, dtype=torch.float)
        x = torch.tensor(self._onehot_nucleotide(sequence), dtype=torch.float)
        batch = torch.zeros(len(sequence), dtype=torch.long)

        return x, edge_index, edge_attr, batch

    @staticmethod
    def _dotbracket_to_edges(dotbracket: str):
        """Convert dot-bracket to edge list with features."""
        # Handle duplex notation
        if "+" in dotbracket:
            strand_break = dotbracket.find("+")
            dotbracket = dotbracket.replace("+", "")
        else:
            strand_break = -1

        N = len(dotbracket)

        # Backbone edges (5' -> 3')
        if strand_break == -1:
            edge_5p = [[i, i + 1] for i in range(N - 1)]
        else:
            edge_5p = [
                [i, i + 1] for i in range(N - 1) if i != strand_break - 1 and i + 1 != strand_break
            ]

        # Hydrogen bond edges (base pairs)
        edge_hbond = []
        flag3p = N - 1
        for i, c in enumerate(dotbracket):
            if c == "(":
                for j in range(flag3p, i, -1):
                    if dotbracket[j] == ")":
                        edge_hbond.append([i, j])
                        flag3p = j - 1
                        break

        # Combine: 5'->3', 3'->5', H-bonds (bidirectional)
        edge_list = (
            edge_5p + [e[::-1] for e in edge_5p] + edge_hbond + [e[::-1] for e in edge_hbond]
        )

        # Edge features: [5'->3', 3'->5', H-bond]
        n_backbone = len(edge_5p)
        n_hbond = len(edge_hbond)
        edge_attr = np.zeros((len(edge_list), 3), dtype=np.float32)
        edge_attr[:n_backbone, 0] = 1  # 5' -> 3'
        edge_attr[n_backbone : 2 * n_backbone, 1] = 1  # 3' -> 5'
        edge_attr[-2 * n_hbond :, 2] = 1  # H-bonds

        return edge_list, edge_attr

    @staticmethod
    def _onehot_nucleotide(sequence: str) -> np.ndarray:
        """One-hot encode nucleotide sequence."""
        mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
        N = len(sequence)
        encoded = np.zeros((N, 4), dtype=np.float32)
        for i, nt in enumerate(sequence.upper()):
            if nt in mapping:
                encoded[i, mapping[nt]] = 1
        return encoded

    def _unnormalize(self, dH_norm: float, Tm_norm: float) -> tuple[float, float]:
        """Unnormalize predictions to real values."""
        dH = dH_norm * (self.sumstats["dH_max"] - self.sumstats["dH_min"]) + self.sumstats["dH_min"]
        Tm = Tm_norm * (self.sumstats["Tm_max"] - self.sumstats["Tm_min"]) + self.sumstats["Tm_min"]
        return dH, Tm

    def predict(
        self,
        sequence: str,
        structure: str,
        na_conc: float = 1.0,
        mg_conc: float = 0.0,
    ) -> GNNPredictionResult:
        """
        Predict thermodynamic parameters using GNN.

        Args:
            sequence: DNA sequence (ATCG only)
            structure: Dot-bracket structure
            na_conc: Sodium concentration in M (for salt adjustment)
            mg_conc: Magnesium concentration in M (for salt adjustment)

        Returns:
            GNNPredictionResult with dH, Tm, dG_37, etc.
        """
        torch, _ = _import_torch()

        # Prepare input
        x, edge_index, edge_attr, batch = self._prepare_input(sequence, structure)

        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        batch = batch.to(self.device)

        # Predict
        with torch.no_grad():
            out = self.model(x, edge_index, edge_attr, batch)

        dH_norm = out[0].item()
        Tm_norm = out[1].item()

        # Unnormalize
        dH, Tm = self._unnormalize(dH_norm, Tm_norm)

        # Calculate dG_37 and dS
        T_ref = 273.15 + 37
        if abs(dH) > 0.1 and Tm > -200:
            dS = dH / (Tm + 273.15)  # kcal/(mol·K)
            dG_37 = dH - T_ref * dS
        else:
            dS = 0.0
            dG_37 = 0.0

        # GC content
        gc = (sequence.upper().count("G") + sequence.upper().count("C")) / len(sequence) * 100

        # Salt adjustment (Na+ and/or Mg2+)
        Tm_adjusted = None
        n_bp = len([c for c in structure if c == "("])
        if na_conc != 1.0 or mg_conc > 0:
            Tm_adjusted = get_salt_adjusted_tm(
                Tm, gc, Na=na_conc, Mg=mg_conc, from_Na=1.0, n_bp=n_bp if n_bp > 1 else None
            )

        return GNNPredictionResult(
            sequence=sequence,
            structure=structure,
            dH=dH,
            Tm=Tm,
            dG_37=dG_37,
            dS=dS,
            Tm_adjusted=Tm_adjusted,
            gc_content=gc,
            prediction_method="gnn",
        )

    def predict_batch(
        self,
        sequences: list[str],
        structures: list[str],
        na_conc: float = 1.0,
        mg_conc: float = 0.0,
        batch_size: int = 64,
    ) -> list[GNNPredictionResult]:
        torch, _ = _import_torch()
        from torch_geometric.data import Data, Batch

        results = []
        n_samples = len(sequences)

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_seqs = sequences[batch_start:batch_end]
            batch_structs = structures[batch_start:batch_end]

            data_list = []
            for seq, struct in zip(batch_seqs, batch_structs):
                x, edge_index, edge_attr, _ = self._prepare_input(seq, struct)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(data)

            batch = Batch.from_data_list(data_list).to(self.device)

            with torch.no_grad():
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            out = out.view(-1, 2)

            for i, (seq, struct) in enumerate(zip(batch_seqs, batch_structs)):
                dH_norm = out[i, 0].item()
                Tm_norm = out[i, 1].item()
                dH, Tm = self._unnormalize(dH_norm, Tm_norm)

                T_ref = 273.15 + 37
                if abs(dH) > 0.1 and Tm > -200:
                    dS = dH / (Tm + 273.15)
                    dG_37 = dH - T_ref * dS
                else:
                    dS, dG_37 = 0.0, 0.0

                gc = (seq.upper().count("G") + seq.upper().count("C")) / len(seq) * 100

                Tm_adjusted = None
                n_bp = len([c for c in struct if c == "("])
                if na_conc != 1.0 or mg_conc > 0:
                    Tm_adjusted = get_salt_adjusted_tm(
                        Tm, gc, Na=na_conc, Mg=mg_conc, from_Na=1.0, n_bp=n_bp if n_bp > 1 else None
                    )

                results.append(
                    GNNPredictionResult(
                        sequence=seq,
                        structure=struct,
                        dH=dH,
                        Tm=Tm,
                        dG_37=dG_37,
                        dS=dS,
                        Tm_adjusted=Tm_adjusted,
                        gc_content=gc,
                    )
                )

        return results


class GNNPredictor:
    """
    High-level GNN predictor interface.

    Wrapper around GNNModel with lazy loading and convenience methods.
    """

    _instance: Optional[GNNModel] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize GNN predictor.

        Args:
            model_path: Path to pretrained weights (uses default if None)
            device: 'cpu' or 'cuda' (auto-detected if None)
        """
        self.model_path = model_path
        self.device = device
        self._model: Optional[GNNModel] = None

    @property
    def model(self) -> GNNModel:
        """Lazy-load the model."""
        if self._model is None:
            self._model = GNNModel(
                model_path=self.model_path,
                device=self.device,
            )
        return self._model

    def predict(
        self,
        sequence: str,
        structure: Optional[str] = None,
        na_conc: float = 1.0,
        mg_conc: float = 0.0,
    ) -> GNNPredictionResult:
        """
        Predict Tm for a DNA hairpin.

        Args:
            sequence: DNA sequence
            structure: Dot-bracket (inferred if not provided)
            na_conc: Sodium concentration in M
            mg_conc: Magnesium concentration in M

        Returns:
            GNNPredictionResult
        """
        if structure is None:
            from ivt_hairpinstat.core.structure_parser import DNAStructureParser

            parser = DNAStructureParser()
            structure = parser.get_target_structure(sequence)

        return self.model.predict(sequence, structure, na_conc, mg_conc)

    def __call__(
        self,
        sequence: str,
        structure: Optional[str] = None,
        na_conc: float = 1.0,
        mg_conc: float = 0.0,
    ) -> GNNPredictionResult:
        return self.predict(sequence, structure, na_conc, mg_conc)


def is_gnn_available() -> bool:
    """Check if GNN dependencies are installed."""
    try:
        _import_torch()
        return True
    except ImportError:
        return False
