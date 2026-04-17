/**
 * iss-locations.html — Locations List Page
 *
 * Retrieves the uploaded CSV from sessionStorage, posts it to the Flask API
 * at /process-csv-json, and renders a card for each ISS location sorted by
 * anomaly score (highest first).
 *
 * Session storage keys used:
 *   uploadedFile   — base64-encoded CSV set by the upload page (index.html)
 *   locationsData  — JSON array of processed location objects, cached here so
 *                    navigating back from the detail page doesn't re-run the
 *                    (slow) KADAIF + MaAsLin analysis
 *
 * Each location card shows the KADAIF anomaly score and a red "Anomaly" badge
 * if the score exceeds 0.80. Clicking a card stores the full location object
 * in sessionStorage and navigates to location-detail.html.
 */

/**
 * Decode the base64-encoded CSV from sessionStorage back into a File object.
 *
 * @returns {File|null} The reconstructed File, or null if none is stored
 */
const getUploadedFile = () => {
    const fileData = sessionStorage.getItem('uploadedFile');
    if (!fileData) return null;

    try {
        const data = JSON.parse(fileData);
        // Decode base64 → binary string → Uint8Array → Blob → File
        const byteCharacters = atob(data.content);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'text/csv' });
        return new File([blob], data.name, { type: 'text/csv' });
    } catch (error) {
        console.error('Error parsing file data:', error);
        return null;
    }
};

/**
 * Send the CSV file to the Flask backend and return the parsed JSON response.
 *
 * POSTs to /process-csv-json which runs KADAIF + MaAsLin3 and returns all
 * locations with their anomaly scores and species lists. This call can take
 * 30–120 seconds depending on dataset size and hardware.
 *
 * @param {File} file - The CSV file to process
 * @returns {Promise<Object>} Parsed JSON: { success, locations, total_locations }
 * @throws {Error} If the server returns an error or the request fails
 */
const processFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:5000/process-csv-json', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process file');
    }

    return await response.json();
};

/**
 * Return the current threshold value from the slider (default 0.80).
 *
 * @returns {number}
 */
const getThreshold = () => {
    const slider = document.getElementById('thresholdSlider');
    return slider ? parseFloat(slider.value) : 0.80;
};

/**
 * Build a location card DOM element.
 *
 * Each card displays:
 *   - Location name (module + location, spaces instead of underscores)
 *   - Number of genus/species found and anomalies at current threshold
 *   - KADAIF anomaly score with a proportional score bar
 *   - "Anomaly" badge if score >= threshold
 *
 * Clicking the card stores the full location object in sessionStorage
 * (so location-detail.html can render it without another API call) and
 * navigates to the detail page.
 *
 * @param {Object} location - Location object from the API response
 * @returns {HTMLElement} The rendered card element
 */
const createLocationCard = (location) => {
    const threshold = getThreshold();

    // Count species whose anomaly score meets the current threshold
    const speciesList = location.species_list || location.species || [];
    const anomaliesCount = speciesList.filter(s => s.anomaly_score >= threshold).length;

    const card = document.createElement('div');
    card.className = 'location-card normal';

    // Cap the score bar at 100% width even if score somehow exceeds 1.0
    const scorePercentage = Math.min(location.anomaly_score * 100, 100);

    card.innerHTML = `
        <div class="location-icon">
            📍
        </div>

        <div class="location-info">
            <div class="location-name">${location.name}</div>

            <div class="location-stats">
                <div class="stat">
                    <span class="stat-value">${location.genus_species_count}</span>
                    <span>genus/species</span>
                </div>
                <div class="stat">
                    <span class="stat-value">${anomaliesCount}</span>
                    <span>anomalies detected</span>
                </div>
            </div>

            <div class="anomaly-score-section">
                <span class="score-label">Anomaly Score:</span>
                <span class="score-value">${location.anomaly_score.toFixed(3)}</span>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: ${scorePercentage}%"></div>
                </div>
            </div>
        </div>

        <div class="arrow-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
    `;

    card.addEventListener('click', () => {
        // Pass the full location object and current threshold to the detail page
        sessionStorage.setItem('selectedLocation', JSON.stringify(location));
        sessionStorage.setItem('anomalyThreshold', String(getThreshold()));
        window.location.href = 'location-detail.html';
    });

    return card;
};

/**
 * Render all location cards into the page and initialize the filter panel.
 *
 * Stores the full locations list in allLocations so filters can be re-applied
 * without re-fetching from the API.
 *
 * @param {Array} locations - Array of location objects from the API
 */
const displayLocations = (locations) => {
    allLocations = locations;
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('thresholdPanel').style.display = 'block';
    applyFilters();
    initFilterPanel(getAllSpeciesNames(locations));
};

/**
 * Show the error state panel with a message.
 *
 * @param {string} message - Error message to display
 */
const showError = (message) => {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
};

/**
 * Page initialization.
 *
 * Two paths:
 *   1. Returning from the detail page — use cached locationsData to avoid
 *      re-running the analysis (which can take over a minute).
 *   2. Fresh load — retrieve the uploaded file from sessionStorage, send it
 *      to the Flask API, display results, and cache them.
 */
const init = async () => {
    // If we already have processed results cached, use them directly
    const cachedLocations = sessionStorage.getItem('locationsData');
    if (cachedLocations) {
        try {
            displayLocations(JSON.parse(cachedLocations));
            return;
        } catch (error) {
            console.error('Error parsing cached locations:', error);
            // Fall through to reprocess if cache is corrupt
        }
    }

    const file = getUploadedFile();
    if (!file) {
        showError('No file found. Please upload a CSV file first.');
        return;
    }

    try {
        const result = await processFile(file);

        if (result.success && result.locations) {
            displayLocations(result.locations);
            // Cache results so navigating back doesn't re-run the analysis
            sessionStorage.setItem('locationsData', JSON.stringify(result.locations));
            // File is no longer needed after processing
            sessionStorage.removeItem('uploadedFile');
        } else {
            showError('Failed to process file. Please try again.');
        }
    } catch (error) {
        showError(error.message || 'An error occurred while processing the file.');
    }
};

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

// All locations from the last processed file — used to re-apply filters
let allLocations = [];

// Active filters: array of { type: 'include'|'exclude', species: string }
let activeFilters = [];

/**
 * Build a deduplicated, sorted list of all species names across all locations.
 * Used to power the autocomplete suggestions dropdown.
 *
 * @param {Array} locations - Full location list from the API
 * @returns {string[]} Sorted array of unique species names
 */
const getAllSpeciesNames = (locations) => {
    const names = new Set();
    locations.forEach(loc => {
        (loc.species_list || []).forEach(s => names.add(s.name));
    });
    return [...names].sort();
};

/**
 * Check whether a location passes all active filters.
 * Include filters require the species to be present (count > 0).
 * Exclude filters require the species to be absent.
 * Matching is case-insensitive and partial (substring).
 *
 * @param {Object} location - Location object with species_list
 * @returns {boolean} True if the location passes all filters
 */
const locationPassesFilters = (location) => {
    const speciesNames = (location.species_list || []).map(s => s.name.toLowerCase());

    for (const filter of activeFilters) {
        const term = filter.species.toLowerCase();
        const hasMatch = speciesNames.some(name => name.includes(term));

        if (filter.type === 'include' && !hasMatch) return false;
        if (filter.type === 'exclude' && hasMatch) return false;
    }
    return true;
};

/**
 * Re-render the locations list based on the current active filters.
 * Shows/hides the no-results panel and updates the filter summary count.
 */
const applyFilters = () => {
    const filtered = allLocations.filter(locationPassesFilters);
    const container = document.getElementById('locationsContainer');
    const noResults = document.getElementById('noResults');
    const summary = document.getElementById('filterSummary');

    container.innerHTML = '';

    if (filtered.length === 0) {
        container.style.display = 'none';
        noResults.style.display = 'block';
    } else {
        noResults.style.display = 'none';
        container.style.display = 'flex';
        filtered.forEach(loc => container.appendChild(createLocationCard(loc)));
    }

    if (activeFilters.length > 0) {
        summary.textContent = `Showing ${filtered.length} of ${allLocations.length} locations`;
    } else {
        summary.textContent = '';
    }
};

/**
 * Add a filter tag to the UI and re-apply filters.
 *
 * @param {'include'|'exclude'} type
 * @param {string} species - Species name or partial name to filter on
 */
const addFilter = (type, species) => {
    const term = species.trim();
    if (!term) return;

    // Don't add duplicate filters
    const exists = activeFilters.some(f => f.type === type && f.species.toLowerCase() === term.toLowerCase());
    if (exists) return;

    activeFilters.push({ type, species: term });
    renderFilterTags();
    applyFilters();
};

/**
 * Remove a filter by index and re-apply.
 *
 * @param {number} index - Index into activeFilters array
 */
const removeFilter = (index) => {
    activeFilters.splice(index, 1);
    renderFilterTags();
    applyFilters();
};

/** Remove all active filters and restore the full location list. */
const clearAllFilters = () => {
    activeFilters = [];
    renderFilterTags();
    applyFilters();
};

/**
 * Re-render the active filter tags row from the current activeFilters array.
 */
const renderFilterTags = () => {
    const container = document.getElementById('activeFilters');
    container.innerHTML = '';

    activeFilters.forEach((filter, index) => {
        const tag = document.createElement('div');
        tag.className = `filter-tag ${filter.type}`;
        tag.innerHTML = `
            <span class="filter-tag-label">${filter.type}</span>
            <span>${filter.species}</span>
            <button class="filter-tag-remove" title="Remove filter">×</button>
        `;
        tag.querySelector('.filter-tag-remove').addEventListener('click', () => removeFilter(index));
        container.appendChild(tag);
    });
};

/**
 * Set up all filter panel interactivity — buttons, input, and suggestions dropdown.
 * Called once after locations are first loaded.
 *
 * @param {string[]} speciesNames - All unique species names for autocomplete
 */
const initFilterPanel = (speciesNames) => {
    const panel = document.getElementById('filterPanel');
    const input = document.getElementById('filterInput');
    const suggestions = document.getElementById('filterSuggestions');
    const includeBtn = document.getElementById('includeBtn');
    const excludeBtn = document.getElementById('excludeBtn');

    panel.style.display = 'block';

    // Show matching suggestions as the user types
    input.addEventListener('input', () => {
        const term = input.value.trim().toLowerCase();
        suggestions.innerHTML = '';

        if (!term) {
            suggestions.classList.remove('visible');
            return;
        }

        const matches = speciesNames.filter(name => name.toLowerCase().includes(term)).slice(0, 10);

        if (matches.length === 0) {
            suggestions.classList.remove('visible');
            return;
        }

        matches.forEach(name => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            // Bold the matching portion of the name
            const idx = name.toLowerCase().indexOf(term);
            item.innerHTML =
                name.slice(0, idx) +
                `<span class="match">${name.slice(idx, idx + term.length)}</span>` +
                name.slice(idx + term.length);
            item.addEventListener('mousedown', (e) => {
                e.preventDefault(); // prevent input blur before click registers
                input.value = name;
                suggestions.classList.remove('visible');
            });
            suggestions.appendChild(item);
        });

        suggestions.classList.add('visible');
    });

    // Hide suggestions when input loses focus
    input.addEventListener('blur', () => {
        setTimeout(() => suggestions.classList.remove('visible'), 150);
    });

    // Add filter on Enter key
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
            addFilter('include', input.value.trim());
            input.value = '';
            suggestions.classList.remove('visible');
        }
    });

    includeBtn.addEventListener('click', () => {
        if (input.value.trim()) {
            addFilter('include', input.value.trim());
            input.value = '';
            suggestions.classList.remove('visible');
        }
    });

    excludeBtn.addEventListener('click', () => {
        if (input.value.trim()) {
            addFilter('exclude', input.value.trim());
            input.value = '';
            suggestions.classList.remove('visible');
        }
    });
};

document.addEventListener('DOMContentLoaded', () => {
    // Restore the last-used threshold (defaults to 0.80 on first visit)
    const slider = document.getElementById('thresholdSlider');
    const thresholdDisplay = document.getElementById('thresholdValue');

    if (slider && thresholdDisplay) {
        const saved = localStorage.getItem('anomalyThreshold');
        if (saved !== null) {
            slider.value = saved;
            thresholdDisplay.textContent = parseFloat(saved).toFixed(2);
        }

        slider.addEventListener('input', () => {
            const val = parseFloat(slider.value).toFixed(2);
            thresholdDisplay.textContent = val;
            localStorage.setItem('anomalyThreshold', val);
            if (allLocations.length > 0) applyFilters();
        });
    }

    init();
});

// Conditional exports for Jest — has no effect in the browser where `module` is undefined
if (typeof module !== 'undefined') {
    module.exports = {
        getUploadedFile, processFile, createLocationCard, displayLocations, showError,
        getAllSpeciesNames, locationPassesFilters, addFilter, removeFilter,
        clearAllFilters, applyFilters
    };
}
