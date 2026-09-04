// Canvas and Drawing State
const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const brushSizeInput = document.getElementById('brushSize');
const brushSizeVal = document.getElementById('brushSizeVal');

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let uploadedFile = null;

// Initialize probability grid
function initProbGrid() {
    const probGrid = document.getElementById('probGrid');
    probGrid.innerHTML = '';
    for (let i = 0; i <= 9; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.id = `prob-row-${i}`;
        row.innerHTML = `
            <span class="prob-label">${i}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" id="prob-fill-${i}"></div>
            </div>
            <span class="prob-percent" id="prob-val-${i}">0.0%</span>
        `;
        probGrid.appendChild(row);
    }
}

// Reset canvas to pure black
function resetCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Canvas Drawing Events
function setupCanvasEvents() {
    resetCanvas();

    brushSizeInput.addEventListener('input', (e) => {
        brushSizeVal.textContent = `${e.target.value}px`;
    });

    // Mouse Events
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        lastX = e.clientX - rect.left;
        lastY = e.clientY - rect.top;
        drawStroke(lastX, lastY, lastX, lastY);
    });

    window.addEventListener('mouseup', () => {
        if (isDrawing) {
            isDrawing = false;
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        drawStroke(lastX, lastY, currentX, currentY);
        lastX = currentX;
        lastY = currentY;
    });

    // Touch Events for Mobile / Tablet
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        lastX = touch.clientX - rect.left;
        lastY = touch.clientY - rect.top;
        drawStroke(lastX, lastY, lastX, lastY);
    }, { passive: false });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        isDrawing = false;
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        const currentX = touch.clientX - rect.left;
        const currentY = touch.clientY - rect.top;
        drawStroke(lastX, lastY, currentX, currentY);
        lastX = currentX;
        lastY = currentY;
    }, { passive: false });
}

function drawStroke(x1, y1, x2, y2) {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = parseInt(brushSizeInput.value, 10);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function clearCanvas() {
    resetCanvas();
    resetResults();
}

function resetResults() {
    document.getElementById('predictedDigit').textContent = '—';
    document.getElementById('confidenceValue').textContent = '0.00%';
    document.getElementById('statusBadge').textContent = 'Ready';
    document.getElementById('statusBadge').style.color = '#10b981';

    for (let i = 0; i <= 9; i++) {
        const fill = document.getElementById(`prob-fill-${i}`);
        const val = document.getElementById(`prob-val-${i}`);
        const row = document.getElementById(`prob-row-${i}`);
        if (fill) fill.style.width = '0%';
        if (fill) fill.classList.remove('highlight');
        if (val) val.textContent = '0.0%';
        if (row) row.classList.remove('active');
    }
}

// Tab Switching
function switchTab(tab) {
    document.getElementById('tab-draw').classList.toggle('active', tab === 'draw');
    document.getElementById('tab-upload').classList.toggle('active', tab === 'upload');
    document.getElementById('draw-view').classList.toggle('active', tab === 'draw');
    document.getElementById('upload-view').classList.toggle('active', tab === 'upload');
    resetResults();
}

// Predict from Canvas
async function predictCanvas() {
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.textContent = 'Predicting...';
    statusBadge.style.color = '#6366f1';

    try {
        const imageData = canvas.toDataURL('image/png');
        const response = await fetch('/predict-canvas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        const result = await response.json();
        if (result.error) {
            alert(result.error);
            statusBadge.textContent = 'Error';
            statusBadge.style.color = '#ef4444';
            return;
        }

        displayResult(result);
    } catch (err) {
        console.error(err);
        alert("Failed to reach server. Make sure 'python app.py' is running.");
        statusBadge.textContent = 'Failed';
        statusBadge.style.color = '#ef4444';
    }
}

// File Upload Handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = function(event) {
        const preview = document.getElementById('imagePreview');
        preview.src = event.target.result;
        document.getElementById('previewContainer').style.display = 'block';
        document.getElementById('predictUploadBtn').disabled = false;
    };
    reader.readAsDataURL(file);
}

// Drag and drop for upload zone
const dropzone = document.getElementById('dropzone');
['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-active');
    }, false);
});
['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-active');
    }, false);
});
dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file && file.type.startsWith('image/')) {
        document.getElementById('fileInput').files = dt.files;
        handleFileSelect({ target: { files: [file] } });
    }
});

// Predict from Uploaded Image
async function predictUpload() {
    if (!uploadedFile) return;

    const statusBadge = document.getElementById('statusBadge');
    statusBadge.textContent = 'Predicting...';
    statusBadge.style.color = '#6366f1';

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
        const response = await fetch('/predict-upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (result.error) {
            alert(result.error);
            statusBadge.textContent = 'Error';
            statusBadge.style.color = '#ef4444';
            return;
        }

        displayResult(result);
    } catch (err) {
        console.error(err);
        alert("Failed to reach server. Make sure 'python app.py' is running.");
        statusBadge.textContent = 'Failed';
        statusBadge.style.color = '#ef4444';
    }
}

// Display Prediction Results
function displayResult(result) {
    const { digit, confidence, probabilities } = result;

    document.getElementById('predictedDigit').textContent = digit;
    document.getElementById('confidenceValue').textContent = `${confidence.toFixed(2)}%`;

    const statusBadge = document.getElementById('statusBadge');
    statusBadge.textContent = 'Success';
    statusBadge.style.color = '#10b981';

    // Update probabilities
    probabilities.forEach((prob, i) => {
        const fill = document.getElementById(`prob-fill-${i}`);
        const val = document.getElementById(`prob-val-${i}`);
        const row = document.getElementById(`prob-row-${i}`);

        if (fill) {
            fill.style.width = `${Math.max(2, prob)}%`;
            fill.classList.toggle('highlight', i === digit);
        }
        if (val) {
            val.textContent = `${prob.toFixed(1)}%`;
        }
        if (row) {
            row.classList.toggle('active', i === digit);
        }
    });
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    initProbGrid();
    setupCanvasEvents();
});
