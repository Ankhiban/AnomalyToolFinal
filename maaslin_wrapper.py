"""
MaAsLin3 Wrapper — Species Anomaly Contribution Analysis

This module runs MaAsLin3 (via an R subprocess) to determine how much each
microbial species contributes to the KADAIF anomaly score across all ISS locations.

MaAsLin3 fits a linear model per species, using the KADAIF score as the outcome
variable and species abundance as the predictor. The resulting coefficient ('coef')
reflects how strongly a species' abundance correlates with anomaly — a high positive
coefficient means the species tends to be more abundant at anomalous locations.

These global coefficients are then combined with local z-scores and percentile ranks
to produce a per-species anomaly score at each individual location.

Requires:
    - R installed and available on PATH as 'Rscript'
    - MaAsLin3 R package installed
    - run_maaslin3.R script in the same directory as this file
"""

import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats  # used for percentileofscore in get_location_species_scores


def calculate_species_anomaly_scores(abundance_data: pd.DataFrame, kadaif_scores: pd.Series) -> pd.DataFrame:
    """
    Calculate per-species anomaly contribution scores using MaAsLin3.

    Calls run_maaslin_analysis() and raises an error if it fails, so the
    caller always gets either a valid result or a clear exception — no silent
    fallbacks.

    Args:
        abundance_data: DataFrame with locations as rows, species as columns,
                        organism counts as values
        kadaif_scores:  Series indexed by location with KADAIF anomaly scores

    Returns:
        DataFrame with columns: feature, coef, pval, qval
        - feature: species name
        - coef: MaAsLin3 linear model coefficient (higher = more anomaly-associated)
        - pval: individual p-value
        - qval: FDR-adjusted q-value

    Raises:
        RuntimeError: If MaAsLin3 analysis fails or returns no results
    """
    print("Running MaAsLin3 analysis...")
    results = run_maaslin_analysis(abundance_data, kadaif_scores)
    if results is None or len(results) == 0:
        raise RuntimeError(
            "MaAsLin3 returned no results. "
            "Ensure R and MaAsLin3 are installed and configured correctly."
        )
    print(f"MaAsLin3 analysis successful - {len(results)} species analyzed")
    return results


def run_maaslin_analysis(abundance_data: pd.DataFrame, kadaif_scores: pd.Series) -> pd.DataFrame:
    """
    Invoke MaAsLin3 via an R subprocess and parse the output.

    Writes abundance data and KADAIF scores to a temporary directory, calls
    run_maaslin3.R with Rscript, then reads the output TSV/CSV back into Python.
    The temporary directory and all its contents are automatically cleaned up
    when the function returns.

    MaAsLin3 output format (TSV):
        feature, metadata, coef, stderr, pval_individual, qval_individual,
        model, [other columns]

    Only 'abundance' model rows are kept (MaAsLin3 also fits a prevalence model).

    Args:
        abundance_data: DataFrame (locations × species) with organism counts
        kadaif_scores:  Series (location → KADAIF score)

    Returns:
        DataFrame with columns [feature, coef, pval, qval], or None if the
        R script fails, times out, or produces no output file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        abundance_file = os.path.join(tmpdir, "abundance.csv")
        metadata_file = os.path.join(tmpdir, "metadata.csv")
        output_dir = os.path.join(tmpdir, "output")

        # Align abundance and score data to the same set of locations
        common_samples = abundance_data.index.intersection(kadaif_scores.index)
        abundance_subset = abundance_data.loc[common_samples]
        scores_subset = kadaif_scores.loc[common_samples]

        # Write inputs for the R script
        abundance_subset.to_csv(abundance_file)
        pd.DataFrame({'KADAIF_Score': scores_subset}).to_csv(metadata_file)

        script_path = Path(__file__).parent / "run_maaslin3.R"

        try:
            result = subprocess.run(
                ["Rscript", str(script_path), abundance_file, metadata_file, output_dir],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout — MaAsLin3 can be slow on large datasets
            )

            if result.returncode != 0:
                print(f"R script failed (exit {result.returncode}):\n{result.stderr}")
                return None

            # MaAsLin3 primary output is a TSV
            results_file = os.path.join(output_dir, "all_results.tsv")
            if os.path.exists(results_file):
                results = pd.read_csv(results_file, sep='\t')

                # MaAsLin3 runs both abundance and prevalence models — keep abundance only
                if 'model' in results.columns:
                    results = results[results['model'] == 'abundance']

                # Normalize column names to the format the rest of the app expects
                results = results.rename(columns={
                    'pval_individual': 'pval',
                    'qval_individual': 'qval'
                })

                if 'feature' in results.columns and 'coef' in results.columns:
                    return results[['feature', 'coef', 'pval', 'qval']].dropna(subset=['coef'])
                return results

            # Some MaAsLin versions write CSV instead of TSV
            results_file_csv = os.path.join(output_dir, "all_results.csv")
            if os.path.exists(results_file_csv):
                return pd.read_csv(results_file_csv)

        except subprocess.TimeoutExpired:
            print("MaAsLin3 analysis timed out after 5 minutes")
            return None
        except FileNotFoundError:
            print("'Rscript' not found — R does not appear to be installed or is not on PATH")
            return None

    return None


def get_location_species_scores(
    location_id: str,
    abundance_data: pd.DataFrame,
    kadaif_scores: pd.Series,
    global_species_scores: pd.DataFrame = None
) -> list:
    """
    Compute a per-species anomaly score for a single ISS location.

    Combines two signals for each species present at the location:
      1. Local signal — how unusual is this species' count at this location
         compared to all other locations? (z-score + percentile rank)
      2. Global signal — how strongly does this species correlate with anomaly
         across all locations? (MaAsLin3 coefficient)

    Final score formula:
        anomaly_score = (percentile × 0.6 + sigmoid(z_score / 2) × 0.4)
                        × importance_weight

    Where importance_weight = (|global_coef| + 1) / 2, scaling from 0.5 to 1.
    Species with no MaAsLin3 result get a weight of 0.5 (score is halved).
    All scores are clipped to [0, 1].

    Args:
        location_id:           Location key matching abundance_data's index
        abundance_data:        DataFrame (locations × species) with organism counts
        kadaif_scores:         Series (location → KADAIF score), used if global
                               scores need to be recomputed
        global_species_scores: Pre-computed MaAsLin3 results DataFrame. If None,
                               calculate_species_anomaly_scores() is called.

    Returns:
        List of dicts sorted by anomaly_score descending, each with keys:
            name, anomaly_score, count, z_score, global_importance
        Returns [] if the location is not found or has no species present.
    """
    if location_id not in abundance_data.index:
        return []

    location_data = abundance_data.loc[location_id]
    present_species = location_data[location_data > 0]

    if len(present_species) == 0:
        return []

    if global_species_scores is None:
        global_species_scores = calculate_species_anomaly_scores(abundance_data, kadaif_scores)

    species_scores = []

    for species_name, count in present_species.items():
        # Look up this species' global importance from MaAsLin3 results
        global_coef = 0.0
        if global_species_scores is not None and len(global_species_scores) > 0:
            match = global_species_scores[global_species_scores['feature'] == species_name]
            if len(match) > 0:
                global_coef = match['coef'].values[0]

        # Z-score: how many standard deviations above/below the mean for this species
        species_mean = abundance_data[species_name].mean()
        species_std = abundance_data[species_name].std()
        z_score = (count - species_mean) / species_std if species_std > 0 else 0.0

        # Sigmoid-normalize the z-score to [0, 1]
        # Dividing by 2 moderates the sigmoid so mid-range z-scores aren't pushed
        # too close to 0 or 1 prematurely
        z_normalized = 1 / (1 + np.exp(-z_score / 2))

        # Scale the MaAsLin3 coefficient to a [0.5, 1] weight
        # Species with no global importance still contribute at half weight
        importance_weight = (abs(global_coef) + 1) / 2

        # Percentile rank of this count among all locations for this species
        percentile = stats.percentileofscore(abundance_data[species_name].values, count) / 100

        # Combine local signals, weighted by global importance
        anomaly_score = np.clip(
            (percentile * 0.6 + z_normalized * 0.4) * importance_weight,
            0, 1
        )

        species_scores.append({
            'name': species_name,
            'anomaly_score': round(float(anomaly_score), 3),
            'count': int(count),
            'z_score': round(float(z_score), 2),
            'global_importance': round(float(global_coef), 3)
        })

    species_scores.sort(key=lambda x: x['anomaly_score'], reverse=True)
    return species_scores


def count_anomalous_species(species_scores: list, threshold: float = 0.8) -> int:
    """
    Count how many species at a location have an anomaly score above a threshold.

    Args:
        species_scores: List of species score dicts from get_location_species_scores()
        threshold:      Anomaly cutoff (default 0.8). Species at or above this
                        value are considered anomalous.

    Returns:
        Integer count of anomalous species
    """
    return sum(1 for s in species_scores if s['anomaly_score'] >= threshold)
