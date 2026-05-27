"""
Anomaly Detection Tool — Flask Backend

This is the main API server for the NASA ISS Microbiome Anomaly Detection Tool.
It accepts CSV uploads of organism count data, runs KADAIF anomaly detection on
each ISS location, and runs MaAsLin3 to score how much each species contributes
to the anomaly at each location.

Endpoints:
    POST /process-csv        — Returns processed data as a downloadable CSV file
    POST /process-csv-json   — Returns processed data as JSON for the frontend UI
    GET  /location-details/<id> — Returns cached species-level detail for one location
    GET  /health             — Health check

Expected input CSV columns: module, loc1, genus, species, Organism_Count, Sample.Collection.Date
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import traceback
from KADAIF import KADAIF
from maaslin_wrapper import (
    calculate_species_anomaly_scores,
    get_location_species_scores,
    count_anomalous_species
)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend (served on a different port)

# In-memory cache storing the most recent analysis run.
# This allows the /location-details endpoint to serve species data without
# reprocessing the CSV on every request. Cleared and rewritten on each new upload.
analysis_cache = {
    'abundance_data': None,       # DataFrame: locations × species counts
    'kadaif_scores': None,        # Series: KADAIF anomaly score per location
    'global_species_scores': None, # DataFrame: MaAsLin3 coefficient per species
    'locations_data': None        # Dict: location_id -> full location result dict
}


def build_species_location_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw organism count records into a locations × species pivot matrix.

    Combines 'module', 'loc1', and 'Sample.Collection.Date' into a single location
    identifier (e.g. 'Node1_Deck_2019-04-01'), combines 'genus' and 'species' into a
    taxon name (e.g. 'Staphylococcus aureus'), then pivots so rows are locations and
    columns are species, with organism counts as values. Including the date ensures
    the same physical location sampled on different dates is treated as a separate row
    rather than being merged together.

    Args:
        df: Raw input DataFrame with columns:
            module, loc1, genus, species, Organism_Count, Sample.Collection.Date

    Returns:
        DataFrame with shape (n_locations, n_species), sorted by both axes
    """
    df = df.copy()

    def normalize_location(s):
        """Strip, collapse internal whitespace, and uppercase for location identifiers."""
        return s.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()

    def normalize_taxon(s):
        """Strip, collapse internal whitespace, and capitalize first letter only."""
        return s.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.capitalize()

    def normalize_date(s):
        """Parse dates and format as YYYY-MM-DD; fall back to raw value (slashes replaced) if unparseable."""
        parsed = pd.to_datetime(s, errors='coerce')
        formatted = parsed.dt.strftime('%Y-%m-%d')
        raw_fallback = s.astype(str).str.strip().str.replace('/', '-', regex=False)
        return formatted.where(formatted.notna(), raw_fallback)

    df['module'] = normalize_location(df['module'])
    df['loc1'] = normalize_location(df['loc1'])
    df['date_key'] = normalize_date(df['Sample.Collection.Date'])

    # Build location key: module[_loc1]_date
    df['module_loc'] = df.apply(
        lambda row: row['module'] if row['loc1'] in ('', 'NAN', 'NONE') else row['module'] + "_" + row['loc1'],
        axis=1
    )
    df['module_loc'] = df['module_loc'] + "_" + df['date_key']

    df['genus_species'] = normalize_taxon(df['genus']) + " " + normalize_taxon(df['species'])

    matrix = df.pivot_table(
        index='genus_species',
        columns='module_loc',
        values='Organism_Count',
        aggfunc='sum',
        fill_value=0
    )

    # Sort both axes for consistency, then transpose so locations are rows
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix.T


@app.route('/process-csv', methods=['POST'])
def process_csv():
    """
    Process an uploaded CSV and return a downloadable CSV with KADAIF scores appended.

    Accepts a multipart/form-data POST with a 'file' field containing the CSV.
    Runs KADAIF on the locations × species matrix and returns the matrix with
    a 'KADAIF_Score' column added, as a downloadable file.

    Returns:
        200: CSV file download (processed_bacteria_analysis.csv)
        400: JSON error if file is missing or required columns are absent
        500: JSON error if processing fails
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        df = pd.read_csv(file)

        required_columns = ['module', 'loc1', 'genus', 'species', 'Organism_Count', 'Sample.Collection.Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'available_columns': list(df.columns)
            }), 400

        species_location_data = build_species_location_matrix(df)

        # Run KADAIF to score each location
        model = KADAIF(
            number_of_trees=100,
            subsample_size=50,
            splitting_method="pcoa",
            normalize=True,
            verbose=True
        )
        anomaly_scores = model.fit_transform(species_location_data)
        species_location_data['KADAIF_Score'] = anomaly_scores.flatten()

        # Return as a downloadable CSV
        output = io.StringIO()
        species_location_data.to_csv(output)
        output.seek(0)
        csv_bytes = io.BytesIO(output.getvalue().encode('utf-8'))
        csv_bytes.seek(0)

        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name='processed_bacteria_analysis.csv'
        )

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/process-csv-json', methods=['POST'])
def process_csv_json():
    """
    Process an uploaded CSV and return full analysis results as JSON.

    This is the primary endpoint used by the frontend UI. It:
      1. Builds the locations × species matrix from the uploaded CSV
      2. Runs KADAIF to produce a per-location anomaly score
      3. Runs MaAsLin3 (via R) to find which species drive anomalies globally
      4. Combines MaAsLin3 coefficients with local z-scores and percentile ranks
         to produce a per-species anomaly score for each location
      5. Caches all results for subsequent /location-details requests
      6. Returns all locations sorted by anomaly score (highest first)

    A location is flagged as an anomaly if its KADAIF score > 0.80.
    A species is flagged as anomalous if its combined score > 0.80.

    Returns:
        200: JSON with keys: success, locations (list), total_locations (int)
        400: JSON error if file is missing or required columns are absent
        500: JSON error if KADAIF or MaAsLin3 analysis fails
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        df = pd.read_csv(file)

        required_columns = ['module', 'loc1', 'genus', 'species', 'Organism_Count', 'Sample.Collection.Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'available_columns': list(df.columns)
            }), 400

        species_location_data = build_species_location_matrix(df)

        # --- Step 1: KADAIF — score each location ---
        model = KADAIF(
            number_of_trees=100,
            subsample_size=50,
            splitting_method="pcoa",
            normalize=True,
            verbose=True
        )
        anomaly_scores = model.fit_transform(species_location_data)
        species_location_data['KADAIF_Score'] = anomaly_scores.flatten()

        # Separate abundance matrix from scores for downstream use
        abundance_only = species_location_data.drop(columns=['KADAIF_Score'])
        kadaif_series = pd.Series(
            species_location_data['KADAIF_Score'].values,
            index=species_location_data.index
        )

        # --- Step 2: MaAsLin3 — score each species globally ---
        # Raises RuntimeError if R/MaAsLin3 is not available or returns no results
        print("Calculating species anomaly scores...")
        global_species_scores = calculate_species_anomaly_scores(abundance_only, kadaif_series)

        # Cache results so /location-details can serve them without reprocessing
        analysis_cache['abundance_data'] = abundance_only
        analysis_cache['kadaif_scores'] = kadaif_series
        analysis_cache['global_species_scores'] = global_species_scores

        # --- Step 3: Build per-location result objects ---
        locations = []
        for location_name in species_location_data.index:
            location_row = species_location_data.loc[location_name]
            kadaif_score = location_row['KADAIF_Score']

            # Get per-species anomaly scores for this location
            species_scores = get_location_species_scores(
                location_name,
                abundance_only,
                kadaif_series,
                global_species_scores
            )

            anomalies_count = count_anomalous_species(species_scores, threshold=0.8)

            # Replace underscores with spaces for display (e.g. 'Node1_Deck' -> 'Node1 Deck')
            display_name = location_name.replace('_', ' ')

            locations.append({
                'id': location_name,
                'name': display_name,
                'genus_species_count': len(species_scores),
                'anomalies_count': anomalies_count,
                'species_list': species_scores,
                'anomaly_score': float(kadaif_score),
                'is_anomaly': float(kadaif_score) > 0.80
            })

        # Sort highest anomaly score first
        locations.sort(key=lambda x: x['anomaly_score'], reverse=True)

        # Cache indexed by location id for O(1) lookup in /location-details
        analysis_cache['locations_data'] = {loc['id']: loc for loc in locations}

        return jsonify({
            'success': True,
            'locations': locations,
            'total_locations': len(locations)
        }), 200

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/location-details/<location_id>', methods=['GET'])
def location_details(location_id):
    """
    Return cached species-level anomaly detail for a single location.

    Must be called after /process-csv-json has been run for the current session,
    as it depends on the in-memory analysis_cache. The location_id should match
    the 'id' field returned by /process-csv-json (module_loc format).

    Args:
        location_id: URL-encoded location identifier (e.g. 'Node1_Deck')

    Returns:
        200: JSON with keys: success, location (full detail dict including species list)
        400: JSON error if no analysis has been run yet
        404: JSON error if location_id is not found in cache
        500: JSON error if an unexpected error occurs
    """
    try:
        if analysis_cache['locations_data'] is None:
            return jsonify({
                'error': 'No analysis data available. Please process a CSV file first.'
            }), 400

        from urllib.parse import unquote
        location_id = unquote(location_id)

        # Try exact match first, then with underscores substituted for spaces
        location_data = analysis_cache['locations_data'].get(location_id)
        if location_data is None:
            location_data = analysis_cache['locations_data'].get(location_id.replace(' ', '_'))

        if location_data is None:
            return jsonify({
                'error': f'Location "{location_id}" not found',
                'available_locations': list(analysis_cache['locations_data'].keys())
            }), 404

        return jsonify({
            'success': True,
            'location': {
                'id': location_data['id'],
                'name': location_data['name'],
                'anomaly_score': location_data['anomaly_score'],
                'is_anomaly': location_data['is_anomaly'],
                'genus_species_count': location_data['genus_species_count'],
                'anomalies_count': location_data['anomalies_count'],
                'species': location_data['species_list']
            }
        }), 200

    except Exception as e:
        print(f"Error getting location details: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error getting location details: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint to confirm the server is running."""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
