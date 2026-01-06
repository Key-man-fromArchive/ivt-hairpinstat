#!/usr/bin/env python3
"""
Coefficient trainer for ivt-thermostat.

Trains linear regression models for dH and dG using the training data
from the dna24 paper. Uses RiboGraphViz for feature extraction.

Usage:
    python scripts/train_coefficients.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Paper data directory
PAPER_DATA_DIR = Path(__file__).parent.parent.parent / "GreenleafLab-nnn_paper-802cd63" / "data"

# Try to import RiboGraphViz, fall back to standalone implementation if not available
HAS_RIBOGRAPHVIZ = False
LoopExtruder = None
StackExtruder = None

try:
    from RiboGraphViz import LoopExtruder, StackExtruder  # type: ignore

    HAS_RIBOGRAPHVIZ = True
except ImportError:
    print("Warning: RiboGraphViz not installed. Using standalone implementation.")


def clean_feature(x: str) -> str:
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
    loops = LoopExtruder(seq_padded, struct_padded, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq_padded, struct_padded, stack_size=stack_size)

    # Clean format
    loops_cleaned = [clean_feature(x) for x in loops]
    stacks_cleaned = [clean_feature(x) for x in stacks]

    # Process hairpin loops
    if sep_base_stack:
        for loop in loops_cleaned.copy():
            seq_part, struct_part = loop.split("_")
            if hp_pattern.match(struct_part):
                # This is a hairpin loop
                # Extract loop without closing bp and base stack separately
                seq_unpadded = seq_part.replace("+", " ")
                hairpin_loop = LoopExtruder(seq_unpadded, struct_part, neighbor_bps=0)[0]
                hairpin_stack = StackExtruder(seq_unpadded, struct_part, stack_size=1)[0]

                # For tetraloops and shorter
                if len(seq_unpadded.replace(" ", "")) <= 6:
                    loops_cleaned.append(clean_feature(hairpin_loop))
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
                    loops_cleaned.append(clean_feature(hairpin_stack))

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


def build_feature_matrix(df: pd.DataFrame, feature_func=None) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature count matrix from dataframe.

    Returns:
        feature_df: DataFrame with feature counts
        all_features: List of all unique features
    """
    if feature_func is None:
        feature_func = extract_features

    # Extract features for each sequence
    all_feature_lists = []
    for _, row in df.iterrows():
        features = feature_func(row["RefSeq"], row["TargetStruct"])
        all_feature_lists.append(features)

    # Get all unique features
    all_features = sorted(set(f for fl in all_feature_lists for f in fl))

    # Build count matrix
    feature_counts = []
    for features in all_feature_lists:
        counts = Counter(features)
        row_counts = [counts.get(f, 0) for f in all_features]
        feature_counts.append(row_counts)

    feature_df = pd.DataFrame(feature_counts, index=df.index, columns=all_features)

    return feature_df, all_features


class LinearRegressionSVD:
    """
    Linear regression using SVD with error weighting.
    Matches the original paper implementation.
    """

    def __init__(self, param: str = "dG_37"):
        self.param = param
        self.coef_ = None
        self.coef_se_ = None
        self.feature_names_ = None
        self.metrics = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        feature_names: list[str],
        singular_value_thresh: float = 1e-15,
    ):
        """
        Fit using weighted least squares via SVD.
        """
        # Weight by inverse error
        A = X / y_err.reshape(-1, 1)
        b = (y / y_err).reshape(-1, 1)

        # SVD solve
        u, s, vh = np.linalg.svd(A, full_matrices=False)

        # Regularize small singular values
        s_inv = 1 / s
        s_inv[s < s[0] * singular_value_thresh] = 0

        # Coefficients and standard errors
        self.coef_ = (vh.T @ np.diag(s_inv) @ u.T @ b).flatten()
        self.coef_se_ = np.sqrt(np.sum((vh.T * s_inv.reshape(1, -1)) ** 2, axis=1))
        self.feature_names_ = feature_names

        # Calculate metrics
        y_pred = X @ self.coef_
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_error = np.sum((y - y_pred) ** 2)

        self.metrics = {
            "rsqr": 1 - ss_error / ss_total,
            "rmse": np.sqrt(ss_error / len(y)),
            "mae": np.mean(np.abs(y - y_pred)),
            "n_samples": len(y),
            "n_features": len(feature_names),
        }

    @property
    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coefficient": self.coef_, "std_error": self.coef_se_}, index=self.feature_names_
        )


class RidgeRegressionWeighted:
    """
    Ridge regression with sample weights (inverse error weighting).
    Better generalization than pure SVD for sparse features.
    """

    def __init__(self, param: str = "dG_37", alpha: float = 1.0):
        self.param = param
        self.alpha = alpha
        self.coef_ = None
        self.coef_se_ = None
        self.feature_names_ = None
        self.metrics = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        feature_names: list[str],
    ):
        """
        Fit using weighted ridge regression.
        """
        # Sample weights (inverse variance)
        weights = 1.0 / (y_err**2)

        # Weighted design matrix
        W = np.diag(np.sqrt(weights))
        A = W @ X
        b = W @ y

        # Ridge regression: (X'WX + alpha*I)^-1 X'Wy
        n_features = X.shape[1]
        XtWX = A.T @ A
        XtWy = A.T @ b

        # Add regularization
        reg_matrix = self.alpha * np.eye(n_features)

        # Solve
        self.coef_ = np.linalg.solve(XtWX + reg_matrix, XtWy)

        # Approximate standard errors (diagonal of inverse)
        try:
            cov_matrix = np.linalg.inv(XtWX + reg_matrix)
            self.coef_se_ = np.sqrt(np.diag(cov_matrix))
        except:
            self.coef_se_ = np.zeros(n_features)

        self.feature_names_ = feature_names

        # Calculate metrics
        y_pred = X @ self.coef_
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_error = np.sum((y - y_pred) ** 2)

        self.metrics = {
            "rsqr": 1 - ss_error / ss_total,
            "rmse": np.sqrt(ss_error / len(y)),
            "mae": np.mean(np.abs(y - y_pred)),
            "n_samples": len(y),
            "n_features": len(feature_names),
        }

    @property
    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coefficient": self.coef_, "std_error": self.coef_se_}, index=self.feature_names_
        )


def train_model(
    train_df: pd.DataFrame,
    param: str,
    feature_func=None,
    use_ridge: bool = True,
    alpha: float = 1.0,
) -> tuple:
    if feature_func is None:
        feature_func = extract_features

    print(f"\nTraining {param} model (Ridge alpha={alpha if use_ridge else 'SVD'})...")

    feature_df, all_features = build_feature_matrix(train_df, feature_func)

    print(f"  Features: {len(all_features)}")
    print(f"  Samples: {len(train_df)}")

    y = train_df[param].values
    y_err = train_df[f"{param}_se"].values

    y_err = np.where(y_err > 0, y_err, np.median(y_err[y_err > 0]))

    valid_mask = ~(np.isnan(y) | np.isnan(y_err))
    X = feature_df.values[valid_mask]
    y = y[valid_mask]
    y_err = y_err[valid_mask]

    print(f"  Valid samples: {len(y)}")

    if use_ridge:
        model = RidgeRegressionWeighted(param=param, alpha=alpha)
    else:
        model = LinearRegressionSVD(param=param)
    model.fit(X, y, y_err, all_features)

    print(f"  R²: {model.metrics['rsqr']:.4f}")
    print(f"  RMSE: {model.metrics['rmse']:.4f}")
    print(f"  MAE: {model.metrics['mae']:.4f}")

    return model, feature_df


def save_coefficients(
    model_dH: LinearRegressionSVD,
    model_dG: LinearRegressionSVD,
    output_path: Path,
):
    """
    Save coefficients to JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "model": "ivt-thermostat",
            "version": "2.0.0",
            "description": "DNA thermodynamics model trained on dna24 data with RiboGraphViz features",
            "dH_metrics": model_dH.metrics,
            "dG_metrics": model_dG.metrics,
        },
        "dH": {
            "coefficients": {
                name: float(coef) for name, coef in zip(model_dH.feature_names_, model_dH.coef_)
            },
            "standard_errors": {
                name: float(se) for name, se in zip(model_dH.feature_names_, model_dH.coef_se_)
            },
        },
        "dG": {
            "coefficients": {
                name: float(coef) for name, coef in zip(model_dG.feature_names_, model_dG.coef_)
            },
            "standard_errors": {
                name: float(se) for name, se in zip(model_dG.feature_names_, model_dG.coef_se_)
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nCoefficients saved to: {output_path}")
    print(f"  dH features: {len(data['dH']['coefficients'])}")
    print(f"  dG features: {len(data['dG']['coefficients'])}")


def validate_model(
    model_dH: LinearRegressionSVD,
    model_dG: LinearRegressionSVD,
    val_df: pd.DataFrame,
    feature_func=None,
):
    """
    Validate model on held-out data.
    """
    if feature_func is None:
        feature_func = extract_features

    print("\n" + "=" * 50)
    print("Validation Results")
    print("=" * 50)

    errors = []
    missing_features_count = 0

    for _, row in val_df.iterrows():
        seq = row["RefSeq"]
        struct = row["TargetStruct"]
        actual_tm = row["Tm"]

        # Extract features
        features = feature_func(seq, struct)

        # Check for missing features
        missing = [f for f in features if f not in model_dH.coef_df.index]
        if missing:
            missing_features_count += 1

        # Predict dH and dG (only use features that exist in model)
        pred_dH = sum(
            model_dH.coef_df.loc[f, "coefficient"] for f in features if f in model_dH.coef_df.index
        )
        pred_dG = sum(
            model_dG.coef_df.loc[f, "coefficient"] for f in features if f in model_dG.coef_df.index
        )

        # Calculate Tm: Tm = T_ref / (1 - dG/dH) - 273.15
        if pred_dH != 0 and pred_dG != pred_dH:
            T_ref = 273.15 + 37
            pred_tm = T_ref / (1 - pred_dG / pred_dH) - 273.15
        else:
            pred_tm = np.nan

        if not np.isnan(pred_tm) and not np.isnan(actual_tm):
            errors.append(pred_tm - actual_tm)

    if errors:
        errors = np.array(errors)
        print(f"N samples evaluated: {len(errors)}")
        if missing_features_count > 0:
            print(f"Samples with missing features: {missing_features_count}")
        print(f"RMSE: {np.sqrt(np.mean(errors**2)):.2f} °C")
        print(f"MAE: {np.mean(np.abs(errors)):.2f} °C")
        print(f"Bias: {np.mean(errors):.2f} °C")

        within_05 = np.mean(np.abs(errors) <= 0.5) * 100
        within_1 = np.mean(np.abs(errors) <= 1.0) * 100
        within_2 = np.mean(np.abs(errors) <= 2.0) * 100

        print(f"Within 0.5°C: {within_05:.1f}%")
        print(f"Within 1.0°C: {within_1:.1f}%")
        print(f"Within 2.0°C: {within_2:.1f}%")

        if within_05 >= 90:
            print("\n✓ TARGET ACHIEVED: >=90% within 0.5°C")
        elif within_1 >= 90:
            print("\n○ GOOD: 90% within 1.0°C")
        else:
            print("\n✗ NEEDS IMPROVEMENT")

        return {
            "rmse": np.sqrt(np.mean(errors**2)),
            "mae": np.mean(np.abs(errors)),
            "bias": np.mean(errors),
            "within_05": within_05,
            "within_1": within_1,
            "within_2": within_2,
        }
    return None


def main():
    """Main entry point."""
    print("=" * 60)
    print("Ivt-Thermostat Coefficient Trainer")
    print("=" * 60)

    if HAS_RIBOGRAPHVIZ:
        print("Using RiboGraphViz for feature extraction")
    else:
        print("Using standalone feature extraction (limited)")

    # Load training data
    train_file = PAPER_DATA_DIR / "models" / "processed" / "arr_v1_1M_n=27732.csv"

    if not train_file.exists():
        print(f"Error: Training data not found at {train_file}")
        sys.exit(1)

    print(f"\nLoading training data from: {train_file}")
    df = pd.read_csv(train_file, index_col=0)
    print(f"Total samples: {len(df)}")

    # Load train/test split
    split_file = PAPER_DATA_DIR / "models" / "raw" / "combined_data_split.json"
    if split_file.exists():
        with open(split_file) as f:
            split_data = json.load(f)
        train_ids = set(split_data.get("train_ind", []))
        test_ids = set(split_data.get("test_ind", []))

        train_df = df[df.index.isin(train_ids)]
        test_df = df[df.index.isin(test_ids)]

        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
    else:
        # Use 80/20 split
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.8
        train_df = df[mask]
        test_df = df[~mask]
        print(f"Train samples: {len(train_df)} (80%)")
        print(f"Test samples: {len(test_df)} (20%)")

    model_dH, _ = train_model(train_df, "dH", use_ridge=False)
    model_dG, _ = train_model(train_df, "dG_37", use_ridge=False)

    # Save coefficients
    output_path = Path(__file__).parent.parent / "data" / "coefficients" / "dna24_coefficients.json"
    save_coefficients(model_dH, model_dG, output_path)

    # Validate on test set
    if len(test_df) > 0:
        validate_model(model_dH, model_dG, test_df)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
