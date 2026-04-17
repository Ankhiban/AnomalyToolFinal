/**
 * index.html — Upload Page
 *
 * Handles CSV file selection (click-to-browse and drag-and-drop) and submission.
 *
 * On submit, the file is NOT sent to the server immediately. Instead it is read
 * into memory, base64-encoded, and stored in sessionStorage so that the
 * iss-locations.html page can retrieve it and send it to the Flask API.
 * This avoids passing the file through URL parameters and keeps the upload
 * experience fast regardless of file size.
 *
 * Required CSV columns: module, loc1, genus, species, Organism_Count
 */

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const chooseFileBtn = document.getElementById('chooseFileBtn');
const submitBtn = document.getElementById('submitBtn');
const fileNameDisplay = document.getElementById('fileName');

let selectedFile = null;

// Open the file picker when the "Choose File" button is clicked.
// stopPropagation prevents the uploadArea click handler from also firing.
chooseFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

// Clicking anywhere else in the upload area also opens the file picker
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Handle file chosen via the file picker dialog
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Highlight the drop zone while a file is being dragged over it
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

// Remove highlight when the drag leaves the drop zone
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

// Handle file dropped onto the upload area
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

/**
 * Validate and register a file selection.
 * Only CSV files are accepted. Updates the UI and enables the submit button.
 *
 * @param {File} file - The file chosen by the user
 */
function handleFile(file) {
    if (!file) return;

    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    selectedFile = file;
    fileNameDisplay.textContent = `Selected: ${file.name}`;
    submitBtn.disabled = false;
}

/**
 * Submit handler — encode the selected file as base64 and store it in
 * sessionStorage, then navigate to the locations page.
 *
 * sessionStorage is used (instead of passing the file directly) because
 * the file needs to survive a page navigation. The locations page retrieves
 * it, sends it to the Flask API, and clears it from storage after processing.
 */
submitBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select a CSV file first');
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';

    try {
        const reader = new FileReader();

        reader.onload = function(e) {
            // Encode binary file content as base64 so it can be stored as a JSON string
            const base64Content = btoa(e.target.result);
            sessionStorage.setItem('uploadedFile', JSON.stringify({
                name: selectedFile.name,
                content: base64Content
            }));
            window.location.href = 'iss-locations.html';
        };

        reader.onerror = function() {
            throw new Error('Failed to read file');
        };

        // readAsBinaryString is used (not readAsDataURL) because btoa() expects
        // a raw binary string, not a data URL with a MIME prefix
        reader.readAsBinaryString(selectedFile);

    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit & Analyze';
    }
});

// Submit button starts disabled until a valid file is selected
submitBtn.disabled = true;

// Conditional exports for Jest — has no effect in the browser where `module` is undefined
if (typeof module !== 'undefined') {
    module.exports = { handleFile };
}
