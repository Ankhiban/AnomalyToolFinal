/**
 * location-detail.html — Location Detail Page
 *
 * Displays the species-level breakdown for a single ISS location.
 * The selected location object is passed from iss-locations.html via
 * sessionStorage ('selectedLocation' key), so no additional API call is needed.
 *
 * On load, species-classifications.csv is fetched to build a lookup table
 * mapping species names to their ecological classification (e.g. "human associated",
 * "environmental", "pathogen"). This drives the colored badge shown next to
 * each species in the table.
 *
 * Each species row shows:
 *   - Species name
 *   - Anomaly score (0–1) with a proportional score bar
 *   - Classification badge (color-coded by type, from the CSV)
 */

/**
 * Retrieve and parse the selected location from sessionStorage.
 *
 * @returns {Object|null} The location object, or null if not found / parse fails
 */
const getSelectedLocation = () => {
    const locationData = sessionStorage.getItem('selectedLocation');
    if (!locationData) return null;

    try {
        return JSON.parse(locationData);
    } catch (error) {
        console.error('Error parsing location data:', error);
        return null;
    }
};

// In-memory lookup: { "staphylococcus aureus": "human associated", ... }
// Populated by loadClassifications() before the species table is rendered.
let speciesClassificationMap = {};

/**
 * Fetch and parse species-classifications.csv into the lookup map.
 *
 * CSV format (no quoting required, but quotes are stripped if present):
 *   Classification,genus_species
 *   human associated,Staphylococcus aureus
 *   environmental,Acinetobacter pittii
 *   ...
 *
 * Keys are stored lowercase for case-insensitive matching. Trailing spaces
 * in either column are trimmed.
 *
 * Errors (file not found, network failure) are caught and logged — the page
 * will still render, but all species will show no classification badge.
 */
const loadClassifications = async () => {
    try {
        const response = await fetch('species-classifications.csv');
        const text = await response.text();
        const lines = text.trim().split('\n');

        // Skip the header row
        for (let i = 1; i < lines.length; i++) {
            const [classification, genusSpecies] = lines[i]
                .split(',')
                .map(s => s.trim().replace(/^"|"$/g, '')); // strip surrounding quotes

            if (genusSpecies && classification) {
                speciesClassificationMap[genusSpecies.toLowerCase()] = classification;
            }
        }
    } catch (error) {
        console.error('Failed to load species-classifications.csv:', error);
    }
};

/**
 * Look up a species name in the classification map and return badge data.
 *
 * The CSS class is derived from the classification string by lowercasing and
 * replacing all non-alphanumeric character runs with hyphens, e.g.:
 *   "human associated (oral)" → "human-associated-oral"
 *
 * This must match the CSS class names defined in location-detail-styles.css.
 *
 * @param {string} speciesName - Full genus + species name (e.g. "Staphylococcus aureus")
 * @returns {{ type: string, label: string }} CSS class name and display label,
 *          or { type: 'unknown', label: '' } if the species is not in the CSV
 */
const getSpeciesStatus = (speciesName) => {
    const classification = speciesClassificationMap[speciesName.toLowerCase()];
    if (!classification) {
        return { type: 'unknown', label: '' };
    }
    // Sanitize to a valid CSS class name: lowercase, non-alphanumeric runs → hyphens
    const type = classification.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    return { type, label: classification };
};

/**
 * Build a single species row element for the table.
 *
 * Each row contains:
 *   - Species name (left column)
 *   - Anomaly score number + proportional bar (middle column)
 *   - Classification badge (right column, hidden if no classification found)
 *
 * @param {Object} species - Species object from the location's species_list:
 *   { name, anomaly_score, count, z_score, global_importance }
 * @returns {HTMLElement} The rendered row div
 */
const createSpeciesRow = (species) => {
    const row = document.createElement('div');
    const status = getSpeciesStatus(species.name);
    const threshold = parseFloat(
        sessionStorage.getItem('anomalyThreshold') ??
        localStorage.getItem('anomalyThreshold') ??
        0.8
    );
    row.className = `species-row${species.anomaly_score >= threshold ? ' pathogen-alert' : ''}`;

    const scorePercentage = Math.min(species.anomaly_score * 100, 100);

    row.innerHTML = `
        <div class="species-name">${species.name}</div>
        <div class="col-score-content">
            <span class="species-score">${species.anomaly_score.toFixed(3)}</span>
            <div class="score-bar-container">
                <div class="score-bar" style="width: ${scorePercentage}%"></div>
            </div>
        </div>
        <div class="col-status">
            ${status.label ? `<span class="status-badge ${status.type}">${status.label}</span>` : ''}
        </div>
    `;

    return row;
};

/**
 * Populate the page with data for the given location.
 *
 * Updates the header, summary line, KADAIF score badge, anomaly tag, and
 * species table. Hides the loading spinner and shows the table when done.
 *
 * @param {Object} location - Full location object from sessionStorage
 */
const displayLocationDetails = (location) => {
    document.getElementById('locationName').textContent = location.name;

    // Compute anomalies count dynamically from the current threshold
    const threshold = parseFloat(
        sessionStorage.getItem('anomalyThreshold') ??
        localStorage.getItem('anomalyThreshold') ??
        0.8
    );
    const speciesList = location.species_list || location.species || [];
    const anomaliesCount = speciesList.filter(s => s.anomaly_score >= threshold).length;

    document.getElementById('locationSummary').innerHTML =
        `<span class="highlight">${location.genus_species_count}</span> genus/species found - ` +
        `<span class="highlight">${anomaliesCount}</span> anomalies detected`;

    document.getElementById('scoreValue').textContent = location.anomaly_score.toFixed(3);

    // Score badge is always shown in neutral style — only species are flagged as anomalies
    document.getElementById('scoreBadge').classList.add('normal');
    document.getElementById('anomalyTag').style.display = 'none';

    // Render species rows — species_list is pre-sorted by anomaly_score descending
    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = '';
    speciesList.forEach(species => tableBody.appendChild(createSpeciesRow(species)));

    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('speciesTable').style.display = 'block';
};

/**
 * Show an alert and redirect to the locations list.
 *
 * @param {string} message - Error message to display
 */
const showError = (message) => {
    alert(message);
    window.location.href = 'iss-locations.html';
};

/**
 * Page initialization.
 *
 * Loads the classification CSV and the location data in sequence — the CSV
 * must be loaded before rendering rows so badges appear immediately.
 */
const init = async () => {
    const location = getSelectedLocation();
    if (!location) {
        showError('No location selected. Please select a location from the list.');
        return;
    }

    document.title = `${location.name} - Bacteria Analysis`;

    // Load classifications first so badges are ready when rows are rendered
    await loadClassifications();

    displayLocationDetails(location);
};

document.addEventListener('DOMContentLoaded', init);

// Conditional exports for Jest — has no effect in the browser where `module` is undefined
if (typeof module !== 'undefined') {
    module.exports = {
        getSelectedLocation,
        loadClassifications,
        getSpeciesStatus,
        createSpeciesRow,
        displayLocationDetails,
        speciesClassificationMap: () => speciesClassificationMap
    };
}
