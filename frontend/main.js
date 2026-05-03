/* ============================================================
   MyLesion — Frontend Logic
   Handles: drag-drop, camera, parallel API calls,
            results rendering, and scan history.
   ============================================================ */

//const API = 'http://localhost:8000';
const API_BASE = 'https://thedominators2026.onrender.com';

let currentFile = null;
let cameraStream = null;
let backendOnline = false;

// ── DOM refs ──────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSec = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const analyzeBtn = document.getElementById('analyze-btn');
const analyzeLabel = document.getElementById('analyze-label');
const resultsSection = document.getElementById('results');
const historySec = document.getElementById('history');
const demoNotice = document.getElementById('demo-notice');
const tabUpload = document.getElementById('tab-upload');
const tabCamera = document.getElementById('tab-camera');
const panelUpload = document.getElementById('panel-upload');
const panelCamera = document.getElementById('panel-camera');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const snapBtn = document.getElementById('snap-btn');
const cameraStop = document.getElementById('camera-stop');
const changeBtn = document.getElementById('change-btn');
const newScanBtn = document.getElementById('new-scan-btn');
const clearHistory = document.getElementById('clear-history');

// Metadata inputs
const inputAge = document.getElementById('input-age');
const inputGender = document.getElementById('input-gender');
const inputSkinType = document.getElementById('input-skin-type');
const inputLoc = document.getElementById('input-localization');
const inputDuration = document.getElementById('input-duration');
const inputItchy = document.getElementById('input-itchy');
const inputPainful = document.getElementById('input-painful');
const inputRaised = document.getElementById('input-raised');

// Round age to nearest 5
inputAge.addEventListener('change', () => {
    if (inputAge.value === '') return;
    const val = parseFloat(inputAge.value);
    inputAge.value = Math.round(val / 5) * 5;
});

// Editor DOM
let cropper = null;
const editorModal = document.getElementById('editor-modal');
const editorImg = document.getElementById('editor-img');
const editorClose = document.getElementById('editor-close');
const editorCancel = document.getElementById('editor-cancel');
const editorApply = document.getElementById('editor-apply');
const adjBrightness = document.getElementById('adj-brightness');
const adjContrast = document.getElementById('adj-contrast');
const adjSaturation = document.getElementById('adj-saturation');
const btnRotate = document.getElementById('btn-rotate');
const btnResetAdj = document.getElementById('btn-reset-adj');

// ── Particles ────────────────────────────────────────────────
(function spawnParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 18; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const size = 4 + Math.random() * 10;
        p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      animation-duration:${12 + Math.random() * 20}s;
      animation-delay:${Math.random() * -20}s;
    `;
        container.appendChild(p);
    }
})();

// ── Backend health check ──────────────────────────────────────
async function checkBackend() {
    try {
        const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
        const data = await r.json();
        backendOnline = r.ok;
        if (!backendOnline || data.demo_mode) showDemoNotice();
        else hideDemoNotice();
    } catch {
        backendOnline = false;
        showDemoNotice();
    }
}
function showDemoNotice() { demoNotice.classList.remove('hidden'); }
function hideDemoNotice() { demoNotice.classList.add('hidden'); }

// ── Tab switching ────────────────────────────────────────────
tabUpload.addEventListener('click', () => switchTab('upload'));
tabCamera.addEventListener('click', () => switchTab('camera'));

function switchTab(tab) {
    tabUpload.classList.toggle('active', tab === 'upload');
    tabCamera.classList.toggle('active', tab === 'camera');
    panelUpload.classList.toggle('hidden', tab !== 'upload');
    panelCamera.classList.toggle('hidden', tab !== 'camera');
    if (tab === 'camera') startCamera();
    else stopCamera();
}

// ── Camera ───────────────────────────────────────────────────
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        cameraVideo.srcObject = cameraStream;
    } catch {
        alert('Camera access denied or unavailable. Please use the upload option.');
        switchTab('upload');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
}

snapBtn.addEventListener('click', () => {
    const w = cameraVideo.videoWidth, h = cameraVideo.videoHeight;
    cameraCanvas.width = w; cameraCanvas.height = h;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, w, h);
    cameraCanvas.toBlob(blob => {
        if (!blob) return;
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        stopCamera();
        switchTab('upload');
        handleFile(file);
    }, 'image/jpeg', 0.92);
});

cameraStop.addEventListener('click', () => { stopCamera(); switchTab('upload'); });

// ── Drag & Drop ──────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});

function handleFile(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) { alert('Image must be under 10 MB.'); return; }
    openEditor(file);
}

// ── Image Editor Logic ────────────────────────────────────────
function openEditor(file) {
    const url = URL.createObjectURL(file);
    editorImg.src = url;
    editorModal.classList.remove('hidden');

    if (cropper) cropper.destroy();

    resetAdjustments();

    cropper = new Cropper(editorImg, {
        aspectRatio: 1, // SCIN training uses square images
        viewMode: 1,
        autoCropArea: 1,
        background: false,
        ready() {
            applyAdjustments();
        }
    });
}

function resetAdjustments() {
    adjBrightness.value = 100;
    adjContrast.value = 100;
    adjSaturation.value = 100;
    applyAdjustments();
}

function applyAdjustments() {
    if (!cropper) return;
    const b = adjBrightness.value;
    const c = adjContrast.value;
    const s = adjSaturation.value;
    // Apply CSS filters to the cropper container for real-time preview
    const container = document.querySelector('.cropper-container .cropper-view-box img');
    const canvas = document.querySelector('.cropper-container .cropper-canvas img');
    const filter = `brightness(${b}%) contrast(${c}%) saturate(${s}%)`;
    if (container) container.style.filter = filter;
    if (canvas) canvas.style.filter = filter;
}

[adjBrightness, adjContrast, adjSaturation].forEach(el => {
    el.addEventListener('input', applyAdjustments);
});

btnRotate.addEventListener('click', () => {
    if (cropper) cropper.rotate(90);
});

btnResetAdj.addEventListener('click', resetAdjustments);

editorCancel.addEventListener('click', closeEditor);
editorClose.addEventListener('click', closeEditor);

function closeEditor() {
    editorModal.classList.add('hidden');
    if (cropper) {
        cropper.destroy();
        cropper = null;
    }
}

editorApply.addEventListener('click', () => {
    if (!cropper) return;

    const canvas = cropper.getCroppedCanvas({
        maxWidth: 1024,
        maxHeight: 1024,
        imageSmoothingQuality: 'high'
    });

    // Apply adjustments to the final canvas
    const ctx = canvas.getContext('2d');
    const b = adjBrightness.value;
    const c = adjContrast.value;
    const s = adjSaturation.value;

    // Create a temporary canvas to apply filters
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext('2d');

    tCtx.filter = `brightness(${b}%) contrast(${c}%) saturate(${s}%)`;
    tCtx.drawImage(canvas, 0, 0);

    tempCanvas.toBlob(blob => {
        currentFile = new File([blob], 'processed-skin.jpg', { type: 'image/jpeg' });
        setPreview(URL.createObjectURL(blob));
        closeEditor();
    }, 'image/jpeg', 0.9);
});

function setPreview(url) {
    previewImg.src = url;
    previewSec.classList.remove('hidden');
    analyzeBtn.disabled = false;
}

changeBtn.addEventListener('click', resetUpload);
function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewSec.classList.add('hidden');
    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    // Reset inputs
    adjBrightness.value = 100;
    adjContrast.value = 100;
    adjSaturation.value = 100;
    inputAge.value = '';
    inputGender.value = '';
    inputSkinType.value = '';
    inputLoc.value = '';
    inputDuration.value = '';
    inputItchy.checked = false;
    inputPainful.checked = false;
    inputRaised.checked = false;
}

// ── Analyze ──────────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);
newScanBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    window.scrollTo({ top: document.getElementById('scanner').offsetTop - 80, behavior: 'smooth' });
});

async function runAnalysis() {
    if (!currentFile) return;
    analyzeBtn.disabled = true;
    analyzeLabel.textContent = 'Analyzing…';
    dropZone.classList.add('scanning');

    showLoaders();
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Run CNN first
    const cnnResult = await fetchCNN(currentFile);
    renderCNN(cnnResult);

    // Then run Gemini with the CNN results
    const geminiResult = await fetchGemini(cnnResult);
    renderGemini(geminiResult);

    saveToHistory(currentFile, cnnResult);
    renderHistory();

    dropZone.classList.remove('scanning');
    analyzeBtn.disabled = false;
    analyzeLabel.textContent = 'Analyze Skin';
}

// ── API calls ────────────────────────────────────────────────
async function fetchCNN(file) {
    if (!backendOnline) return null;
    try {
        const fd = new FormData();
        fd.append('file', file);
        // Optional metadata
        if (inputAge.value) fd.append('age', inputAge.value);
        if (inputGender.value) fd.append('gender', inputGender.value);
        if (inputSkinType.value) fd.append('skin_type', inputSkinType.value);
        if (inputLoc.value) fd.append('localization', inputLoc.value);
        if (inputDuration.value) fd.append('duration', inputDuration.value);
        fd.append('is_itchy', inputItchy.checked);
        fd.append('is_painful', inputPainful.checked);
        fd.append('is_raised', inputRaised.checked);

        const r = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd });
        return await r.json();
    } catch { return null; }
}

async function fetchGemini(cnnData = null) {
    if (!backendOnline) return null;
    try {
        const fd = new FormData();
        if (cnnData) fd.append('cnn_result', JSON.stringify(cnnData));

        // Metadata for Gemini
        if (inputAge.value) fd.append('age', inputAge.value);
        if (inputGender.value) fd.append('gender', inputGender.value);
        if (inputSkinType.value) fd.append('skin_type', inputSkinType.value);
        if (inputLoc.value) fd.append('localization', inputLoc.value);
        if (inputDuration.value) fd.append('duration', inputDuration.value);
        fd.append('is_itchy', inputItchy.checked);
        fd.append('is_painful', inputPainful.checked);
        fd.append('is_raised', inputRaised.checked);

        const r = await fetch(`${API_BASE}/gemini-analyze`, { method: 'POST', body: fd });
        return await r.json();
    } catch { return null; }
}

// ── Loaders ──────────────────────────────────────────────────
function showLoaders() {
    ['cnn-loader', 'gemini-loader'].forEach(id => document.getElementById(id).classList.remove('hidden'));
    ['cnn-content', 'gemini-content'].forEach(id => document.getElementById(id).classList.add('hidden'));
}

// ── Render CNN ───────────────────────────────────────────────
function renderCNN(data) {
    const loader = document.getElementById('cnn-loader');
    const content = document.getElementById('cnn-content');
    loader.classList.add('hidden');
    content.classList.remove('hidden');

    // Use demo if no data
    if (!data || data.error) data = DEMO_CNN;

    document.getElementById('cnn-condition').textContent = data.condition;
    document.getElementById('cnn-common').textContent = data.common_name;
    document.getElementById('cnn-description').textContent = data.description;

    // Severity pill
    const pill = document.getElementById('severity-pill');
    const sevClass = { low: 'sev-low', medium: 'sev-medium', high: 'sev-high' }[data.severity] || 'sev-low';
    pill.className = `severity-pill ${sevClass}`;
    pill.textContent = data.severity_label;

    // Confidence bars
    const barsEl = document.getElementById('confidence-bars');
    barsEl.innerHTML = '';
    (data.top3 || []).forEach((item, i) => {
        const pct = Math.round(item.probability * 100);
        const div = document.createElement('div');
        div.className = 'conf-item';
        div.innerHTML = `
      <div class="conf-header">
        <span class="conf-label">${item.label}</span>
        <span class="conf-pct">${pct}%</span>
      </div>
      <div class="conf-track">
        <div class="conf-fill${i === 0 ? ' top' : ''}" data-pct="${pct}"></div>
      </div>`;
        barsEl.appendChild(div);
    });
    // Animate bars after paint
    requestAnimationFrame(() => {
        document.querySelectorAll('.conf-fill').forEach(el => {
            el.style.width = el.dataset.pct + '%';
        });
    });

    // Seek doctor CTA
    const cta = document.getElementById('seek-doctor-cta');
    cta.classList.toggle('hidden', !data.seek_doctor);

    // Demo tag
    if (data._demo) document.getElementById('demo-cnn-tag').classList.remove('hidden');
}

// ── Render Gemini ─────────────────────────────────────────────
function renderGemini(data) {
    const loader = document.getElementById('gemini-loader');
    const content = document.getElementById('gemini-content');
    loader.classList.add('hidden');
    content.classList.remove('hidden');

    if (!data || data.error || data.parse_error) data = DEMO_GEMINI;

    document.getElementById('gemini-observations').textContent = data.visual_observations || '';
    document.getElementById('gemini-explanation').textContent = data.explanation || '';
    document.getElementById('gemini-disclaimer-text').textContent = data.disclaimer || '';

    // Urgency banner
    const urgencyMap = {
        monitor: { cls: 'u-monitor', icon: '👁️', label: data.urgency_label || 'Monitor' },
        schedule: { cls: 'u-schedule', icon: '📅', label: data.urgency_label || 'Schedule a visit' },
        seek_care: { cls: 'u-seek_care', icon: '🚨', label: data.urgency_label || 'Seek care soon' },
    };
    const urg = urgencyMap[data.urgency] || urgencyMap.monitor;
    const banner = document.getElementById('urgency-banner');
    banner.className = `urgency-banner ${urg.cls}`;
    document.getElementById('urgency-icon').textContent = urg.icon;
    document.getElementById('urgency-label').textContent = urg.label;
    document.getElementById('urgency-reason').textContent = data.urgency_reason || '';

    // Lists
    renderList('care-tips', data.care_tips || []);
    renderList('doctor-questions', data.doctor_questions || []);

    if (data._demo) document.getElementById('demo-gemini-tag').classList.remove('hidden');
}

function renderList(id, items) {
    const ul = document.getElementById(id);
    ul.innerHTML = '';
    items.forEach(text => {
        const li = document.createElement('li');
        li.textContent = text;
        ul.appendChild(li);
    });
}

// ── Scan History ──────────────────────────────────────────────
function saveToHistory(file, cnnData) {
    const reader = new FileReader();
    reader.onload = e => {
        const history = getHistory();
        history.unshift({
            img: e.target.result, condition: cnnData?.condition || 'Unknown',
            severity: cnnData?.severity || 'low', ts: Date.now()
        });
        if (history.length > 5) history.pop();
        localStorage.setItem('mylesion_history', JSON.stringify(history));
    };
    reader.readAsDataURL(file);
}

function getHistory() {
    try { return JSON.parse(localStorage.getItem('mylesion_history')) || []; }
    catch { return []; }
}

function renderHistory() {
    const history = getHistory();
    if (!history.length) return;
    historySec.classList.remove('hidden');
    const grid = document.getElementById('history-grid');
    grid.innerHTML = '';
    history.forEach(item => {
        const card = document.createElement('div');
        card.className = 'history-card';
        card.innerHTML = `
      <img class="history-thumb" src="${item.img}" alt="Scan thumbnail" />
      <div class="history-info">
        <div class="history-condition">${item.condition}</div>
        <div class="history-meta">${new Date(item.ts).toLocaleDateString()}</div>
      </div>`;
        grid.appendChild(card);
    });
}

clearHistory.addEventListener('click', () => {
    localStorage.removeItem('mylesion_history');
    document.getElementById('history-grid').innerHTML = '';
    historySec.classList.add('hidden');
});

// ── Demo data ─────────────────────────────────────────────────
const DEMO_CNN = {
    condition: 'Eczema', common_name: 'Atopic Dermatitis', code: 'Eczema',
    confidence: 0.8924, severity: 'medium', severity_label: 'Moderate Risk',
    seek_doctor: false,
    description: 'Eczema is a condition that makes your skin red and itchy. It is extremely common and often chronic, requiring consistent moisturizing and management.',
    color: 'amber',
    top3: [
        { code: 'Eczema', label: 'Eczema', probability: 0.8924 },
        { code: 'Psoriasis', label: 'Psoriasis', probability: 0.0612 },
        { code: 'Tinea', label: 'Tinea', probability: 0.0214 },
    ],
    _demo: true
};

const DEMO_GEMINI = {
    visual_observations: 'The image shows several small, red inflammatory papules and a few closed comedones (whiteheads) clustered on the forehead. There is mild surrounding erythema but no deep cystic lesions.',
    likely_condition: 'Acne Vulgaris',
    explanation: 'This appears to be common acne, which occurs when hair follicles become plugged with oil and dead skin cells. It is a very common condition, especially during hormonal shifts or periods of high oil production.',
    urgency: 'monitor', urgency_label: 'Keep an eye on it',
    urgency_reason: 'Condition is common and currently shows only mild inflammation with no signs of severe infection.',
    care_tips: [
        'Wash your face twice a day with a gentle, non-comedogenic cleanser.',
        'Avoid picking or squeezing the spots, as this can lead to scarring or infection.',
        'Consider over-the-counter treatments containing benzoyl peroxide or salicylic acid.',
    ],
    doctor_questions: [
        'Are these breakouts related to any recent changes in my skincare routine or diet?',
        'Would a prescription-strength retinoid be appropriate for my skin type?',
        'How can I prevent post-inflammatory hyperpigmentation (dark spots) after these heal?',
    ],
    disclaimer: 'This AI analysis is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.',
    _demo: true
};

// ── Init ──────────────────────────────────────────────────────
checkBackend();
renderHistory();
