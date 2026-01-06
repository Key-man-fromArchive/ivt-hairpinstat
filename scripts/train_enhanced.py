#!/usr/bin/env python3
"""
Enhanced coefficient trainer with structure-specific models and direct Tm regression.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

PAPER_DATA_DIR = Path(__file__).parent.parent.parent / "GreenleafLab-nnn_paper-802cd63" / "data"

HAS_RIBOGRAPHVIZ = False
LoopExtruder = None
StackExtruder = None

try:
    from RiboGraphViz import LoopExtruder, StackExtruder

    HAS_RIBOGRAPHVIZ = True
except ImportError:
    print("Warning: RiboGraphViz not installed.")


def clean_feature(x: str) -> str:
    cleaned = x.replace(" ", "+").replace(",", "_")
    if len(cleaned) > 4 and cleaned[1] == "y" and cleaned.split("_")[1] == "((+))":
        cleaned = f"x{cleaned[4]}+{cleaned[0]}y_((+))"
    return cleaned


def extract_features_rgv(seq: str, struct: str, stack_size: int = 2) -> list[str]:
    hp_pattern = re.compile(r"^\([.]+\)")
    pad = stack_size - 1

    seq_padded = "x" * pad + seq + "y" * pad
    struct_padded = "(" * pad + struct + ")" * pad

    loops = LoopExtruder(seq_padded, struct_padded, neighbor_bps=1)
    stacks = StackExtruder(seq_padded, struct_padded, stack_size=stack_size)

    loops_cleaned = [clean_feature(x) for x in loops]
    stacks_cleaned = [clean_feature(x) for x in stacks]

    for loop in loops_cleaned.copy():
        seq_part, struct_part = loop.split("_")
        if hp_pattern.match(struct_part):
            seq_unpadded = seq_part.replace("+", " ")
            hairpin_loop = LoopExtruder(seq_unpadded, struct_part, neighbor_bps=0)[0]
            hairpin_stack = StackExtruder(seq_unpadded, struct_part, stack_size=1)[0]

            if len(seq_unpadded.replace(" ", "")) <= 6:
                loops_cleaned.append(clean_feature(hairpin_loop))
            else:
                loops_cleaned.append("NNNNN_.....")

            loops_cleaned.append(clean_feature(hairpin_stack))
            loops_cleaned.remove(loop)

        elif struct_part == "(..(+)..)":
            seq_parts = seq_part.split("+")
            if len(seq_parts) == 2:
                s1, s2 = seq_parts
                mm = f"{s1[1:3]}+{s2[1:3]}_..+.."
                mm_stack = f"{s1[0]}+{s1[3]}+{s2[0]}+{s2[3]}_(+(+)+)"
                loops_cleaned.append(mm)
                loops_cleaned.append(mm_stack)
                loops_cleaned.remove(loop)

    feature_list = loops_cleaned + stacks_cleaned
    return [x for x in feature_list if "_" in x]


def classify_structure(struct: str, series: str = "") -> str:
    """Classify structure into categories for model selection."""
    series_lower = series.lower() if series else ""

    # Use series info if available
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

    # Infer from structure
    dot_matches = re.findall(r"\.+", struct)
    if len(dot_matches) == 1:
        loop_size = len(dot_matches[0])
        if loop_size == 3:
            return "triloop"
        elif loop_size == 4:
            return "tetraloop"
        elif loop_size <= 6:
            return "simple_hairpin"

    if len(dot_matches) > 1:
        return "complex"

    return "other"


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    all_feature_lists = []
    for _, row in df.iterrows():
        try:
            features = extract_features_rgv(row["RefSeq"], row["TargetStruct"])
        except Exception:
            features = []
        all_feature_lists.append(features)

    all_features = sorted(set(f for fl in all_feature_lists for f in fl))

    feature_counts = []
    for features in all_feature_lists:
        counts = Counter(features)
        row_counts = [counts.get(f, 0) for f in all_features]
        feature_counts.append(row_counts)

    feature_df = pd.DataFrame(feature_counts, index=df.index, columns=all_features)
    return feature_df, all_features


class LinearRegressionSVD:
    def __init__(self, param: str = "dG_37"):
        self.param = param
        self.coef_ = None
        self.coef_se_ = None
        self.feature_names_ = None
        self.metrics = {}

    def fit(self, X: np.ndarray, y: np.ndarray, y_err: np.ndarray, feature_names: list[str]):
        A = X / y_err.reshape(-1, 1)
        b = (y / y_err).reshape(-1, 1)

        u, s, vh = np.linalg.svd(A, full_matrices=False)

        s_inv = 1 / s
        s_inv[s < s[0] * 1e-15] = 0

        self.coef_ = (vh.T @ np.diag(s_inv) @ u.T @ b).flatten()
        self.coef_se_ = np.sqrt(np.sum((vh.T * s_inv.reshape(1, -1)) ** 2, axis=1))
        self.feature_names_ = feature_names

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


def train_model(df: pd.DataFrame, param: str, name: str = "") -> tuple:
    print(f"\nTraining {param} model{' (' + name + ')' if name else ''}...")

    feature_df, all_features = build_feature_matrix(df)

    print(f"  Features: {len(all_features)}, Samples: {len(df)}")

    y = df[param].values
    y_err_col = f"{param}_se"
    if y_err_col in df.columns:
        y_err = df[y_err_col].values
        y_err = np.where(y_err > 0, y_err, np.nanmedian(y_err[y_err > 0]))
    else:
        y_err = np.ones(len(y)) * 0.5

    valid_mask = ~(np.isnan(y) | np.isnan(y_err))
    X = feature_df.values[valid_mask]
    y = y[valid_mask]
    y_err = y_err[valid_mask]

    if len(y) < 10:
        print(f"  SKIP: Only {len(y)} valid samples")
        return None, None

    model = LinearRegressionSVD(param=param)
    model.fit(X, y, y_err, all_features)

    print(f"  R²: {model.metrics['rsqr']:.4f}, RMSE: {model.metrics['rmse']:.4f}")

    return model, feature_df


def train_direct_tm_model(df: pd.DataFrame, name: str = "") -> tuple:
    """Train a model that directly predicts Tm from features."""
    print(f"\nTraining direct Tm model{' (' + name + ')' if name else ''}...")

    feature_df, all_features = build_feature_matrix(df)

    y = df["Tm"].values
    y_err_col = "Tm_se"
    if y_err_col in df.columns:
        y_err = df[y_err_col].values
        y_err = np.where(y_err > 0, y_err, np.nanmedian(y_err[y_err > 0]))
    else:
        y_err = np.ones(len(y)) * 1.0

    valid_mask = ~(np.isnan(y) | np.isnan(y_err))
    X = feature_df.values[valid_mask]
    y = y[valid_mask]
    y_err = y_err[valid_mask]

    print(f"  Features: {len(all_features)}, Samples: {len(y)}")

    if len(y) < 10:
        print(f"  SKIP: Only {len(y)} valid samples")
        return None, None

    model = LinearRegressionSVD(param="Tm")
    model.fit(X, y, y_err, all_features)

    print(f"  R²: {model.metrics['rsqr']:.4f}, RMSE: {model.metrics['rmse']:.2f} °C")

    return model, feature_df


def validate_tm_model(model, val_df: pd.DataFrame, name: str = ""):
    """Validate direct Tm model."""
    if model is None:
        return None

    print(f"\nValidating Tm model{' (' + name + ')' if name else ''}...")

    errors = []
    for _, row in val_df.iterrows():
        try:
            features = extract_features_rgv(row["RefSeq"], row["TargetStruct"])
        except Exception:
            continue

        pred_tm = sum(
            model.coef_df.loc[f, "coefficient"] for f in features if f in model.coef_df.index
        )
        actual_tm = row["Tm"]

        if not np.isnan(pred_tm) and not np.isnan(actual_tm):
            errors.append(pred_tm - actual_tm)

    if not errors:
        return None

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    within_05 = np.mean(np.abs(errors) <= 0.5) * 100
    within_1 = np.mean(np.abs(errors) <= 1.0) * 100
    within_2 = np.mean(np.abs(errors) <= 2.0) * 100

    print(f"  N={len(errors)}, RMSE={rmse:.2f}°C")
    print(f"  Within 0.5°C: {within_05:.1f}%, 1.0°C: {within_1:.1f}%, 2.0°C: {within_2:.1f}%")

    return {"rmse": rmse, "within_05": within_05, "within_1": within_1, "within_2": within_2}


def save_enhanced_coefficients(models: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "model": "ivt-thermostat-enhanced",
            "version": "3.0.0",
            "description": "Enhanced DNA thermodynamics with structure-specific and direct Tm models",
        },
        "models": {},
    }

    for model_name, model_data in models.items():
        if model_data["model"] is None:
            continue

        model = model_data["model"]
        data["models"][model_name] = {
            "type": model_data["type"],
            "param": model.param,
            "metrics": model.metrics,
            "coefficients": {
                name: float(coef) for name, coef in zip(model.feature_names_, model.coef_)
            },
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nEnhanced coefficients saved to: {output_path}")


def main():
    print("=" * 70)
    print("Enhanced Ivt-Thermostat Trainer")
    print("Structure-specific models + Direct Tm regression")
    print("=" * 70)

    if not HAS_RIBOGRAPHVIZ:
        print("ERROR: RiboGraphViz required for enhanced training")
        sys.exit(1)

    train_file = PAPER_DATA_DIR / "models" / "processed" / "arr_v1_1M_n=27732.csv"
    if not train_file.exists():
        print(f"Error: Training data not found at {train_file}")
        sys.exit(1)

    print(f"\nLoading data from: {train_file}")
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
    else:
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.8
        train_df = df[mask]
        test_df = df[~mask]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Classify structures
    if "Series" in df.columns:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["struct_type"] = train_df.apply(
            lambda r: classify_structure(r["TargetStruct"], r.get("Series", "")), axis=1
        )
        test_df["struct_type"] = test_df.apply(
            lambda r: classify_structure(r["TargetStruct"], r.get("Series", "")), axis=1
        )
    else:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["struct_type"] = train_df["TargetStruct"].apply(lambda s: classify_structure(s))
        test_df["struct_type"] = test_df["TargetStruct"].apply(lambda s: classify_structure(s))

    print("\nStructure type distribution (train):")
    print(train_df["struct_type"].value_counts())

    models = {}

    # 1. Train global dH/dG models
    print("\n" + "=" * 50)
    print("GLOBAL MODELS (dH, dG)")
    print("=" * 50)

    model_dH, _ = train_model(train_df, "dH", "global")
    model_dG, _ = train_model(train_df, "dG_37", "global")

    models["global_dH"] = {"model": model_dH, "type": "thermodynamic"}
    models["global_dG"] = {"model": model_dG, "type": "thermodynamic"}

    # 2. Train global direct Tm model
    print("\n" + "=" * 50)
    print("GLOBAL DIRECT Tm MODEL")
    print("=" * 50)

    model_tm_global, _ = train_direct_tm_model(train_df, "global")
    models["global_Tm"] = {"model": model_tm_global, "type": "direct_Tm"}

    # Validate global Tm model
    validate_tm_model(model_tm_global, test_df, "global")

    # 3. Train structure-specific models
    print("\n" + "=" * 50)
    print("STRUCTURE-SPECIFIC MODELS")
    print("=" * 50)

    structure_types = ["tetraloop", "triloop", "watson_crick", "simple_hairpin"]

    for struct_type in structure_types:
        train_subset = train_df[train_df["struct_type"] == struct_type]
        test_subset = test_df[test_df["struct_type"] == struct_type]

        if len(train_subset) < 50:
            print(f"\nSkipping {struct_type}: only {len(train_subset)} samples")
            continue

        print(f"\n--- {struct_type.upper()} (n={len(train_subset)}) ---")

        # Direct Tm model for this structure type
        model_tm, _ = train_direct_tm_model(train_subset, struct_type)
        models[f"{struct_type}_Tm"] = {"model": model_tm, "type": "direct_Tm"}

        if len(test_subset) > 10:
            validate_tm_model(model_tm, test_subset, struct_type)

    # 4. Train simple hairpin combined model (tetra + tri + watson_crick + simple)
    print("\n" + "=" * 50)
    print("SIMPLE STRUCTURES COMBINED MODEL")
    print("=" * 50)

    simple_types = ["tetraloop", "triloop", "watson_crick", "simple_hairpin"]
    train_simple = train_df[train_df["struct_type"].isin(simple_types)]
    test_simple = test_df[test_df["struct_type"].isin(simple_types)]

    print(f"Simple structures: train={len(train_simple)}, test={len(test_simple)}")

    model_tm_simple, _ = train_direct_tm_model(train_simple, "simple_combined")
    models["simple_Tm"] = {"model": model_tm_simple, "type": "direct_Tm"}

    if len(test_simple) > 10:
        validate_tm_model(model_tm_simple, test_simple, "simple_combined")

    # 5. Save all models
    output_path = Path(__file__).parent.parent / "data" / "coefficients" / "dna24_enhanced.json"
    save_enhanced_coefficients(models, output_path)

    # 6. Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Model Comparison on Test Set")
    print("=" * 70)

    comparison_results = []

    # Compare different models on simple structures
    for model_name in ["global_Tm", "simple_Tm"]:
        if model_name in models and models[model_name]["model"]:
            result = validate_tm_model(models[model_name]["model"], test_simple, model_name)
            if result:
                comparison_results.append((model_name, result))

    # Compare dH/dG derived Tm vs direct Tm
    print("\n--- dH/dG Derived Tm (simple structures) ---")
    if model_dH and model_dG:
        errors = []
        for _, row in test_simple.iterrows():
            try:
                features = extract_features_rgv(row["RefSeq"], row["TargetStruct"])
            except Exception:
                continue

            pred_dH = sum(
                model_dH.coef_df.loc[f, "coefficient"]
                for f in features
                if f in model_dH.coef_df.index
            )
            pred_dG = sum(
                model_dG.coef_df.loc[f, "coefficient"]
                for f in features
                if f in model_dG.coef_df.index
            )

            if pred_dH != 0 and abs(pred_dG / pred_dH) < 1:
                pred_tm = (273.15 + 37) / (1 - pred_dG / pred_dH) - 273.15
                if not np.isnan(pred_tm) and -50 < pred_tm < 150:
                    errors.append(pred_tm - row["Tm"])

        if errors:
            errors = np.array(errors)
            print(f"  N={len(errors)}, RMSE={np.sqrt(np.mean(errors**2)):.2f}°C")
            print(f"  Within 0.5°C: {np.mean(np.abs(errors) <= 0.5) * 100:.1f}%")
            print(f"  Within 1.0°C: {np.mean(np.abs(errors) <= 1.0) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
