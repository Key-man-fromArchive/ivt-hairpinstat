#!/usr/bin/env python3
"""
Extract coefficients from dna24 paper codebase.

This script trains the 1,338 parameter Rich Parameter Model and saves
the coefficients as JSON files for use with ivt-thermostat.

Usage:
    python scripts/extract_coefficients.py

Requirements:
    - Original paper codebase at ../GreenleafLab-nnn_paper-802cd63/
    - All dependencies installed (pandas, numpy, sklearn, etc.)
    - Note: RiboGraphViz and NUPACK required for training
"""

import sys
import os
import json
from pathlib import Path

# Add original codebase to path
PAPER_CODE_DIR = Path(__file__).parent.parent.parent / "GreenleafLab-nnn_paper-802cd63"
sys.path.insert(0, str(PAPER_CODE_DIR))

# Change to paper code directory for relative file paths
os.chdir(PAPER_CODE_DIR)


def train_rich_parameter_model():
    """Train the 1,338 parameter Rich Parameter Model."""
    import numpy as np
    import pandas as pd
    from nnn import util, fileio
    from nnn import motif_fit as mf

    # Load data
    arr_1M = pd.read_csv("./data/models/processed/arr_v1_1M_n=27732.csv", index_col=0)
    data_split_dict = fileio.read_json("./data/models/raw/data_split.json")

    # Config for 1,338 parameter model
    config = dict(
        use_train_set_ratio=1.0,
        secondary_struct="target",
        fit_method="svd",
        feature_method="get_feature_list",
        fit_intercept=True,
        symmetry=True,
        fix_some_coef=True,
        sep_base_stack=True,
        stack_size=2,
    )

    # Fixed parameter classes from NUPACK
    fixed_pclass = [
        "hairpin_size",
        "interior_size",
        "bulge_size",
        "hairpin_triloop",
        "hairpin_tetraloop",
        "terminal_mismatch",
        "stack",
    ]

    # Get features
    feature_kwargs = dict(
        symmetry=config["symmetry"],
        sep_base_stack=True,
        hairpin_mm=False,
        ignore_base_stack=False,
        stack_size=config["stack_size"],
        fit_intercept=config["fit_intercept"],
    )

    feats = mf.get_feature_count_matrix(
        arr_1M,
        feature_method=config["feature_method"],
        feature_style="nnn",
        **feature_kwargs,
    )
    print(f"Feature matrix shape: {feats.shape}")
    print(f"First 5 features: {feats.columns[:5].tolist()}")

    # Get fixed parameters from NUPACK
    from nnn import mupack

    fixed_coef_df, fixed_feature_names = mupack.get_fixed_params(
        param_set_template_file="./models/dna04.json", fixed_pclass=fixed_pclass
    )
    fixed_feature_names = [x for x in fixed_feature_names if x in feats.columns]
    print(f"Number of fixed features: {len(fixed_feature_names)}")

    fix_coef_kwargs = dict(
        fixed_feature_names=fixed_feature_names,
        coef_df=None,
    )

    train_kwargs = dict(
        feats=feats,
        train_only=True,
        method=config["fit_method"],
        use_train_set_ratio=config["use_train_set_ratio"],
        fix_some_coef=config["fix_some_coef"],
        fix_coef_kwargs=fix_coef_kwargs,
    )

    # Train models for dH and dG
    lr_dict = {}
    param_name_dict = {"dH": "dH", "dG": "dG_37"}

    for param in param_name_dict:
        print(f"\nTraining {param} model...")
        train_kwargs["fix_coef_kwargs"]["coef_df"] = fixed_coef_df[[param]]
        lr_dict[param] = mf.fit_param(
            arr_1M, data_split_dict, param=param_name_dict[param], **train_kwargs
        )
        print(f"  MAE: {lr_dict[param].metrics['mae']:.4f}")
        print(f"  R²: {lr_dict[param].metrics['rsqr']:.4f}")

    return lr_dict, feats


def save_coefficients_as_json(lr_dict, output_dir: Path):
    """Save coefficients as JSON files."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    for param_name, lr_model in lr_dict.items():
        coef_df = lr_model.coef_df
        coef_se_df = lr_model.coef_se_df

        # Create coefficient dictionary
        coef_dict = {
            "metadata": {
                "param": param_name,
                "num_features": len(coef_df),
                "metrics": lr_model.metrics,
            },
            "coefficients": {},
            "standard_errors": {},
        }

        # Add coefficients
        for feature_name in coef_df.index:
            coef_value = float(coef_df.loc[feature_name].values[0])
            se_value = float(coef_se_df.loc[feature_name].values[0])

            coef_dict["coefficients"][feature_name] = coef_value
            coef_dict["standard_errors"][feature_name] = se_value

        # Save to JSON
        output_file = output_dir / f"coefficients_{param_name}.json"
        with open(output_file, "w") as f:
            json.dump(coef_dict, f, indent=2)

        print(f"Saved {param_name} coefficients to {output_file}")
        print(f"  Total features: {len(coef_dict['coefficients'])}")


def create_combined_coefficient_file(lr_dict, output_dir: Path):
    """Create a single combined coefficient file for both dH and dG."""
    import json

    combined = {
        "metadata": {
            "model": "dna24_rich_parameter",
            "version": "1.0.0",
            "description": "1,338 parameter Rich Parameter Model from dna24 paper",
            "reference": "Nature Communications 2025",
        },
        "dH": {
            "coefficients": {},
            "standard_errors": {},
            "metrics": lr_dict["dH"].metrics,
        },
        "dG": {
            "coefficients": {},
            "standard_errors": {},
            "metrics": lr_dict["dG"].metrics,
        },
    }

    for param_name in ["dH", "dG"]:
        lr_model = lr_dict[param_name]
        coef_df = lr_model.coef_df
        coef_se_df = lr_model.coef_se_df

        for feature_name in coef_df.index:
            coef_value = float(coef_df.loc[feature_name].values[0])
            se_value = float(coef_se_df.loc[feature_name].values[0])

            combined[param_name]["coefficients"][feature_name] = coef_value
            combined[param_name]["standard_errors"][feature_name] = se_value

    output_file = output_dir / "dna24_coefficients.json"
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved combined coefficients to {output_file}")
    return combined


def main():
    """Main entry point."""
    print("=" * 60)
    print("DNA24 Coefficient Extraction Script")
    print("=" * 60)

    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "coefficients"

    try:
        # Train model
        print("\n[1/3] Training Rich Parameter Model...")
        lr_dict, feats = train_rich_parameter_model()

        # Save individual coefficient files
        print("\n[2/3] Saving individual coefficient files...")
        save_coefficients_as_json(lr_dict, output_dir)

        # Save combined file
        print("\n[3/3] Creating combined coefficient file...")
        combined = create_combined_coefficient_file(lr_dict, output_dir)

        print("\n" + "=" * 60)
        print("SUCCESS: Coefficients extracted and saved!")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")
        print(f"dH coefficients: {len(combined['dH']['coefficients'])}")
        print(f"dG coefficients: {len(combined['dG']['coefficients'])}")

    except ImportError as e:
        print(f"\nERROR: Missing dependency - {e}")
        print("\nThis script requires the original paper's dependencies:")
        print("  - RiboGraphViz (for feature extraction)")
        print("  - NUPACK (for fixed parameters)")
        print("  - wandb (for logging, can be mocked)")
        print("\nConsider using the pre-extracted coefficients if available.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
