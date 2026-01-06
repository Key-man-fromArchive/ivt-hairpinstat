"""CLI for ivt-hairpinstat: DNA hairpin Tm prediction using dna24 Rich Parameter Model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ivt_hairpinstat.core.predictor import HairpinPredictor, SaltConditions
from ivt_hairpinstat.core.thermodynamics import get_gc_content

app = typer.Typer(
    name="ivt-hairpinstat",
    help="DNA hairpin Tm prediction using dna24 Rich Parameter Model. NOTE: For hairpins only, not duplexes.",
    no_args_is_help=True,
)
console = Console()


def get_default_coefficients_path() -> Path:
    base = Path(__file__).parent.parent.parent.parent / "data" / "coefficients"
    enhanced = base / "dna24_enhanced.json"
    standard = base / "dna24_coefficients.json"
    return enhanced if enhanced.exists() else standard


@app.command()
def predict(
    sequence: str = typer.Argument(..., help="DNA sequence (ATCG)"),
    structure: Optional[str] = typer.Option(
        None, "--structure", "-s", help="Dot-bracket structure (inferred if not provided)"
    ),
    sodium: float = typer.Option(
        0.05, "--sodium", "-na", help="Sodium concentration in M (default 50 mM)"
    ),
    magnesium: float = typer.Option(
        0.0,
        "--magnesium",
        "-mg",
        help="Magnesium concentration in M (e.g., 0.002 for 2mM PCR buffer)",
    ),
    dna_conc: Optional[float] = typer.Option(
        None, "--dna-conc", "-c", help="DNA concentration in M (for duplex Tm)"
    ),
    use_thermodynamic: bool = typer.Option(
        False, "--thermo", "-t", help="Use thermodynamic Tm (dH/dG) instead of direct regression"
    ),
    use_gnn: bool = typer.Option(
        False, "--gnn", "-g", help="Use GNN model (~1.8°C MAE, requires torch)"
    ),
    auto_model: bool = typer.Option(
        False, "--auto", "-a", help="Auto-select model: Linear for tri/tetraloops, GNN otherwise"
    ),
    ensemble: bool = typer.Option(
        False, "--ensemble", "-e", help="Ensemble prediction (weighted avg of Linear + GNN)"
    ),
    coefficients: Optional[Path] = typer.Option(
        None, "--coefficients", "-f", help="Path to coefficients JSON file"
    ),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """
    Predict thermodynamic parameters for a DNA hairpin.

    Examples:
        ivt-hairpinstat predict GCGCAAAAGCGC -s "((((....))))"
        ivt-hairpinstat predict GCATGCGAAAGCATGC --sodium 0.1 --json
    """
    coef_path = coefficients or get_default_coefficients_path()

    if not coef_path.exists():
        console.print(f"[red]Error: Coefficients file not found: {coef_path}[/red]")
        raise typer.Exit(1)

    try:
        predictor = HairpinPredictor(
            coefficients_file=coef_path,
            use_direct_tm=not use_thermodynamic,
            use_gnn=use_gnn,
            auto_select_model=auto_model or ensemble,
        )
    except ImportError as e:
        console.print(f"[red]GNN requires torch and torch_geometric: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    salt_conditions = SaltConditions(Na=sodium, Mg=magnesium)

    try:
        if ensemble:
            result = predictor.predict_ensemble(
                sequence=sequence,
                structure=structure,
                salt_conditions=salt_conditions,
            )
        else:
            result = predictor.predict(
                sequence=sequence,
                structure=structure,
                salt_conditions=salt_conditions,
                DNA_conc=dna_conc,
            )
    except Exception as e:
        console.print(f"[red]Prediction error: {e}[/red]")
        raise typer.Exit(1)

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_result_table(result, verbose)


def _print_result_table(result, verbose: bool = False):
    table = Table(title="Hairpin Tm Prediction", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    table.add_row(
        "Sequence", result.sequence[:50] + ("..." if len(result.sequence) > 50 else ""), ""
    )
    table.add_row(
        "Structure", result.structure[:50] + ("..." if len(result.structure) > 50 else ""), ""
    )
    table.add_row("Tm", f"{result.Tm:.1f}", "°C")

    if result.Tm_adjusted:
        table.add_row("Tm (adjusted)", f"{result.Tm_adjusted:.1f}", "°C")

    table.add_row("dH", f"{result.dH:.2f}", "kcal/mol")
    table.add_row("dG (37°C)", f"{result.dG_37:.2f}", "kcal/mol")
    table.add_row("dS", f"{result.dS:.4f}", "kcal/(mol·K)")
    table.add_row("GC Content", f"{result.gc_content:.1f}", "%")
    table.add_row("Confidence", result.confidence or "N/A", "")
    table.add_row("Method", result.prediction_method, "")

    if result.warning:
        table.add_row("Warning", result.warning, "")

    if result.structure_type:
        table.add_row("Structure Type", result.structure_type, "")

    if verbose:
        table.add_row("Features", str(len(result.features)), "")
        table.add_row("Unknown Features", str(result.n_unknown_features), "")

    console.print(table)


@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="Input CSV/TSV file with sequences"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (default: stdout)"
    ),
    sequence_col: str = typer.Option("sequence", "--seq-col", help="Name of sequence column"),
    structure_col: Optional[str] = typer.Option(
        None, "--struct-col", help="Name of structure column"
    ),
    sodium: float = typer.Option(0.05, "--sodium", "-na", help="Sodium concentration in M"),
    magnesium: float = typer.Option(0.0, "--magnesium", "-mg", help="Magnesium concentration in M"),
    use_gnn: bool = typer.Option(False, "--gnn", "-g", help="Use GNN model for all"),
    auto_model: bool = typer.Option(False, "--auto", "-a", help="Auto-select model per structure"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size for GNN"),
    coefficients: Optional[Path] = typer.Option(
        None, "--coefficients", "-f", help="Path to coefficients JSON file"
    ),
):
    import csv

    coef_path = coefficients or get_default_coefficients_path()

    if not coef_path.exists():
        console.print(f"[red]Error: Coefficients file not found: {coef_path}[/red]")
        raise typer.Exit(1)

    try:
        predictor = HairpinPredictor(
            coefficients_file=coef_path,
            use_gnn=use_gnn,
            auto_select_model=auto_model,
        )
    except ImportError as e:
        console.print(f"[red]GNN requires torch and torch_geometric: {e}[/red]")
        raise typer.Exit(1)

    salt_conditions = SaltConditions(Na=sodium, Mg=magnesium)

    delimiter = "\t" if input_file.suffix == ".tsv" else ","

    with open(input_file, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    sequences = [row.get(sequence_col, "") for row in rows]
    structures_raw = [row.get(structure_col) if structure_col else None for row in rows]

    valid_indices = [i for i, seq in enumerate(sequences) if seq]
    valid_seqs = [sequences[i] for i in valid_indices]
    valid_structs: Optional[list[str]] = None
    if structure_col:
        valid_structs = [s for i, s in enumerate(structures_raw) if i in valid_indices and s]

    try:
        pred_results = predictor.predict_batch(
            valid_seqs, valid_structs, salt_conditions, batch_size=batch_size
        )
    except Exception as e:
        console.print(f"[red]Batch prediction error: {e}[/red]")
        raise typer.Exit(1)

    results = []
    pred_idx = 0
    for i, row in enumerate(rows):
        if i in valid_indices:
            r = pred_results[pred_idx]
            pred_idx += 1
            results.append(
                {
                    **row,
                    "Tm_pred": f"{r.Tm:.1f}",
                    "dH_pred": f"{r.dH:.2f}",
                    "dG_37_pred": f"{r.dG_37:.2f}",
                    "method": r.prediction_method,
                    "confidence": r.confidence,
                }
            )
        else:
            results.append(
                {
                    **row,
                    "Tm_pred": "SKIPPED",
                    "dH_pred": "SKIPPED",
                    "dG_37_pred": "SKIPPED",
                    "method": "",
                    "confidence": "empty sequence",
                }
            )

    if output_file:
        with open(output_file, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        console.print(f"[green]Results written to {output_file}[/green]")
    else:
        if results:
            writer = csv.DictWriter(sys.stdout, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


@app.command()
def features(
    sequence: str = typer.Argument(..., help="DNA sequence"),
    structure: str = typer.Argument(..., help="Dot-bracket structure"),
    coefficients: Optional[Path] = typer.Option(
        None, "--coefficients", "-f", help="Path to coefficients JSON file"
    ),
):
    """
    Analyze NNN features in a DNA hairpin and their thermodynamic contributions.

    Example:
        ivt-hairpinstat features GCGCAAAAGCGC "((((....))))"
    """
    coef_path = coefficients or get_default_coefficients_path()

    if not coef_path.exists():
        console.print(f"[red]Error: Coefficients file not found: {coef_path}[/red]")
        raise typer.Exit(1)

    predictor = HairpinPredictor(coefficients_file=coef_path)
    report = predictor.get_feature_contributions(sequence, structure)

    console.print(f"\n[bold]Feature Analysis Report[/bold]\n")
    console.print(f"Sequence:  {report['sequence']}")
    console.print(f"Structure: {report['structure']}")
    console.print(f"Total dH:  {report['total_dH']:.2f} kcal/mol")
    console.print(f"Total dG:  {report['total_dG']:.3f} kcal/mol")
    console.print(f"N features: {report['n_features']}")
    console.print(f"N unknown:  {report['n_unknown']}\n")

    if report["features"]:
        table = Table(title="Feature Contributions", show_header=True)
        table.add_column("Feature")
        table.add_column("dH", justify="right")
        table.add_column("dG", justify="right")
        table.add_column("Known")

        for feat in report["features"]:
            table.add_row(
                feat.get("feature", "N/A"),
                f"{feat.get('dH', 0):.2f}",
                f"{feat.get('dG', 0):.3f}",
                "Y" if feat.get("known") else "N",
            )
        console.print(table)


@app.command()
def info():
    """Show information about the package and loaded model."""
    console.print("\n[bold]Ivt-Hairpinstat[/bold]")
    console.print("DNA hairpin Tm prediction using dna24 models")
    console.print("[yellow]NOTE: This is for hairpin structures only, not duplexes.[/yellow]\n")

    coef_path = get_default_coefficients_path()

    if coef_path.exists():
        try:
            predictor = HairpinPredictor(coefficients_file=coef_path)
            model_info = predictor.get_model_info()

            console.print(f"[bold cyan]Linear Model (Rich Parameter):[/bold cyan]")
            console.print(f"  Coefficients: {coef_path.name}")
            console.print(f"  Version: {model_info.get('version', 'N/A')}")
            console.print(f"  dH features: {model_info.get('n_dH_coefficients', 0)}")
            console.print(f"  dG features: {model_info.get('n_dG_coefficients', 0)}")
            console.print(f"  Direct Tm: {model_info.get('has_direct_tm', False)}")
        except Exception as e:
            console.print(f"[yellow]Error reading coefficients: {e}[/yellow]")
    else:
        console.print(f"[yellow]No coefficients file found at: {coef_path}[/yellow]")

    console.print()
    try:
        from ivt_hairpinstat.core.gnn_predictor import is_gnn_available

        gnn_ok = is_gnn_available()
        if gnn_ok:
            console.print(f"[bold cyan]GNN Model:[/bold cyan]")
            console.print(f"  Status: [green]Available[/green]")
            console.print(f"  Parameters: 287,136")
            console.print(f"  Expected MAE: ~1.8°C")
            console.print(f"  Usage: --gnn flag")
        else:
            console.print(f"[bold cyan]GNN Model:[/bold cyan]")
            console.print(f"  Status: [yellow]Not available[/yellow]")
            console.print(f"  Install: pip install torch torch_geometric")
    except Exception:
        console.print(f"[bold cyan]GNN Model:[/bold cyan] [yellow]Not available[/yellow]")


@app.command()
def validate(
    validation_file: Path = typer.Argument(..., help="Validation CSV with actual Tm values"),
    sequence_col: str = typer.Option("RefSeq", help="Sequence column name"),
    structure_col: str = typer.Option("TargetStruct", help="Structure column name"),
    tm_col: str = typer.Option("Tm", help="Actual Tm column name"),
    coefficients: Optional[Path] = typer.Option(None, "--coefficients", "-f"),
):
    """
    Validate model predictions against experimental data.

    Example:
        ivt-hairpinstat validate test_data.csv --tm-col Tm
    """
    import csv
    import statistics

    coef_path = coefficients or get_default_coefficients_path()

    if not coef_path.exists():
        console.print(f"[red]Error: Coefficients file not found[/red]")
        raise typer.Exit(1)

    predictor = HairpinPredictor(coefficients_file=coef_path)

    delimiter = "\t" if validation_file.suffix == ".tsv" else ","

    with open(validation_file, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    errors = []
    abs_errors = []

    for row in rows:
        seq = row.get(sequence_col, "")
        struct = row.get(structure_col, "")
        actual_tm = float(row.get(tm_col, 0))

        if not seq or not actual_tm:
            continue

        try:
            result = predictor.predict(sequence=seq, structure=struct)
            error = result.Tm - actual_tm
            errors.append(error)
            abs_errors.append(abs(error))
        except Exception:
            continue

    if not errors:
        console.print("[red]No valid predictions made[/red]")
        raise typer.Exit(1)

    rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
    mae = statistics.mean(abs_errors)
    bias = statistics.mean(errors)
    within_05 = sum(1 for e in abs_errors if e <= 0.5) / len(abs_errors) * 100
    within_1 = sum(1 for e in abs_errors if e <= 1.0) / len(abs_errors) * 100
    within_2 = sum(1 for e in abs_errors if e <= 2.0) / len(abs_errors) * 100

    table = Table(title="Validation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("N samples", str(len(errors)))
    table.add_row("RMSE", f"{rmse:.2f} C")
    table.add_row("MAE", f"{mae:.2f} C")
    table.add_row("Bias", f"{bias:.2f} C")
    table.add_row("Within 0.5C", f"{within_05:.1f}%")
    table.add_row("Within 1.0C", f"{within_1:.1f}%")
    table.add_row("Within 2.0C", f"{within_2:.1f}%")

    console.print(table)

    if within_05 >= 90:
        console.print(
            "\n[green bold]SUCCESS: Target accuracy achieved (>=90% within 0.5C)[/green bold]"
        )
    elif within_1 >= 90:
        console.print("\n[yellow]GOOD: 90% within 1.0C[/yellow]")
    else:
        console.print("\n[red]NEEDS IMPROVEMENT: Below target accuracy[/red]")


def main():
    app()


if __name__ == "__main__":
    main()
