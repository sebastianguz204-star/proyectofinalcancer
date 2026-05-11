// ============================================================
// MODEL INFERENCE ENGINE
// ============================================================
let MODEL = null;
let selectedModel = 'lr';

async function loadModels() {
  const res = await fetch('js/model_data.json');
  MODEL = await res.json();
  updateHomeStats();
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function relu(x) { return Math.max(0, x); }

function scale(features) {
  return features.map((v, i) => (v - MODEL.scaler.mean[i]) / MODEL.scaler.std[i]);
}

function predictLR(features) {
  const scaled = scale(features);
  const coef = MODEL.logistic_regression.coef;
  const intercept = MODEL.logistic_regression.intercept;
  const logit = scaled.reduce((s, v, i) => s + v * coef[i], intercept);
  const prob = sigmoid(logit);
  return { prob, label: prob >= 0.5 ? 1 : 0 };
}

function predictNN(features) {
  const scaled = scale(features);
  let a = scaled;
  const layers = MODEL.neural_network.layers;
  for (let l = 0; l < layers.length; l++) {
    const { weights, biases } = layers[l];
    const next = biases.map((b, j) => {
      const z = a.reduce((s, v, i) => s + v * weights[i][j], b);
      return l < layers.length - 1 ? relu(z) : z;
    });
    a = next;
  }
  const prob = sigmoid(a[0]);
  return { prob, label: prob >= 0.5 ? 1 : 0 };
}

function predict(features) {
  return selectedModel === 'lr' ? predictLR(features) : predictNN(features);
}

// ============================================================
// HOME STATS
// ============================================================
function updateHomeStats() {
  if (!MODEL) return;
  const lrAcc = MODEL.logistic_regression.test_metrics.accuracy;
  const nnAcc = MODEL.neural_network.test_metrics.accuracy;
  document.getElementById('stat-lr').textContent = lrAcc + '%';
  document.getElementById('stat-nn').textContent = nnAcc + '%';
  document.getElementById('stat-feat').textContent = MODEL.feature_names.length;
  document.getElementById('stat-samples').textContent = MODEL.logistic_regression.X_test.length;
  document.getElementById('lr-acc-bar').style.width = lrAcc + '%';
  document.getElementById('nn-acc-bar').style.width = nnAcc + '%';
  document.getElementById('lr-acc-num').textContent = lrAcc + '%';
  document.getElementById('nn-acc-num').textContent = nnAcc + '%';
}

// ============================================================
// NAVIGATION
// ============================================================
function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`[data-view="${id}"]`).classList.add('active');
}

// ============================================================
// MODEL SELECTION
// ============================================================
function selectModel(m) {
  selectedModel = m;
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
  document.getElementById('card-' + m).classList.add('selected');
}

// ============================================================
// INDIVIDUAL PREDICTION
// ============================================================
function buildFeatureInputs() {
  if (!MODEL) return;
  const grid = document.getElementById('feature-grid');
  grid.innerHTML = MODEL.feature_names.map((name, i) => `
    <div class="feature-item">
      <label>${name}</label>
      <input type="number" step="any" id="feat-${i}" value="0" placeholder="0.0">
    </div>
  `).join('');
}

function loadSample(type) {
  if (!MODEL) return;
  const Xtest = MODEL.logistic_regression.X_test;
  const ytest = MODEL.logistic_regression.y_test;
  let idx = 0;
  if (type === 'benign') {
    idx = ytest.findIndex(v => v === 1);
  } else {
    idx = ytest.findIndex(v => v === 0);
  }
  const sample = Xtest[idx];
  MODEL.feature_names.forEach((_, i) => {
    const el = document.getElementById('feat-' + i);
    if (el) el.value = sample[i].toFixed(4);
  });
}

function runIndividualPrediction() {
  if (!MODEL) return;
  const features = MODEL.feature_names.map((_, i) => parseFloat(document.getElementById('feat-' + i).value) || 0);
  const result = predict(features);
  showIndividualResult(result);
}

function showIndividualResult({ prob, label }) {
  const card = document.getElementById('individual-result');
  card.classList.add('show');
  const verdict = label === 1 ? 'Benigno' : 'Maligno';
  const cls = label === 1 ? 'benign' : 'malignant';
  const probDisplay = label === 1 ? (prob * 100).toFixed(1) : ((1 - prob) * 100).toFixed(1);

  document.getElementById('res-verdict').textContent = verdict;
  document.getElementById('res-verdict').className = 'result-verdict ' + cls;
  document.getElementById('res-prob').textContent = probDisplay;
  document.getElementById('res-model').textContent = selectedModel === 'lr' ? 'Regresión Logística' : 'Red Neuronal';

  const fill = document.getElementById('res-prob-bar');
  fill.className = 'prob-fill ' + cls;
  fill.style.width = '0';
  setTimeout(() => { fill.style.width = probDisplay + '%'; }, 100);

  const emoji = label === 1 ? '✅' : '⚠️';
  document.getElementById('res-icon').textContent = emoji;

  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================================
// BATCH PREDICTION
// ============================================================
let batchData = null;

function setupUpload() {
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) processCSV(file);
  });
  input.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) processCSV(file);
  });
}

function processCSV(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const lines = e.target.result.trim().split('\n');
    const header = lines[0].split(',').map(s => s.trim().toLowerCase());
    const rows = lines.slice(1).map(l => l.split(',').map(Number));
    batchData = { header, rows };
    runBatchOnData(rows);
  };
  reader.readAsText(file);
}

function useDemoData() {
  if (!MODEL) return;
  const X = MODEL.logistic_regression.X_test.slice(0, 20);
  const y = MODEL.logistic_regression.y_test.slice(0, 20);
  batchData = { header: MODEL.feature_names, rows: X, trueLabels: y };
  runBatchOnData(X, y);
}

function runBatchOnData(rows, trueLabels = null) {
  if (!MODEL) return;

  const bar = document.getElementById('batch-progress');
  bar.parentElement.style.display = 'block';

  const results = [];
  let correct = 0;
  rows.forEach((row, i) => {
    const { prob, label } = predict(row);
    const probFinal = label === 1 ? prob : 1 - prob;
    const trueLbl = trueLabels ? trueLabels[i] : null;
    const isCorrect = trueLbl !== null ? (label === trueLbl) : null;
    if (isCorrect) correct++;
    results.push({ features: row, prob: probFinal, label, trueLbl, isCorrect });
    bar.style.width = ((i + 1) / rows.length * 100) + '%';
  });

  setTimeout(() => {
    bar.parentElement.style.display = 'none';
    bar.style.width = '0';
    renderBatchResults(results, correct, rows.length, trueLabels);
  }, 400);
}

function renderBatchResults(results, correct, total, trueLabels) {
  const section = document.getElementById('batch-results');
  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth' });

  // Table
  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = results.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td><span class="badge ${r.label === 1 ? 'badge-benign' : 'badge-malignant'}">${r.label === 1 ? 'Benigno' : 'Maligno'}</span></td>
      <td>${(r.prob * 100).toFixed(1)}%</td>
      ${r.trueLbl !== null ? `<td><span class="badge ${r.trueLbl === 1 ? 'badge-benign' : 'badge-malignant'}">${r.trueLbl === 1 ? 'Benigno' : 'Maligno'}</span></td>` : '<td>—</td>'}
      ${r.isCorrect !== null ? `<td><span class="badge ${r.isCorrect ? 'badge-correct' : 'badge-wrong'}">${r.isCorrect ? '✓' : '✗'}</span></td>` : '<td>—</td>'}
    </tr>
  `).join('');

  // Confusion Matrix & Metrics (only if true labels available)
  if (trueLabels) {
    const metrics = selectedModel === 'lr' ? MODEL.logistic_regression.test_metrics : MODEL.neural_network.test_metrics;
    // compute from current predictions
    let tp = 0, tn = 0, fp = 0, fn = 0;
    results.forEach(r => {
      if (r.label === 1 && r.trueLbl === 1) tp++;
      else if (r.label === 0 && r.trueLbl === 0) tn++;
      else if (r.label === 1 && r.trueLbl === 0) fp++;
      else fn++;
    });
    document.getElementById('cm-tp').textContent = tp;
    document.getElementById('cm-tn').textContent = tn;
    document.getElementById('cm-fp').textContent = fp;
    document.getElementById('cm-fn').textContent = fn;

    const acc = ((tp + tn) / total * 100).toFixed(1);
    const prec = tp + fp > 0 ? (tp / (tp + fp) * 100).toFixed(1) : '0.0';
    const rec = tp + fn > 0 ? (tp / (tp + fn) * 100).toFixed(1) : '0.0';
    const f1 = prec > 0 || rec > 0 ? (2 * prec * rec / (parseFloat(prec) + parseFloat(rec))).toFixed(1) : '0.0';

    document.getElementById('m-acc').textContent = acc + '%';
    document.getElementById('m-prec').textContent = prec + '%';
    document.getElementById('m-rec').textContent = rec + '%';
    document.getElementById('m-f1').textContent = f1 + '%';

    document.getElementById('matrix-section').style.display = 'block';
  } else {
    document.getElementById('matrix-section').style.display = 'none';
  }
}

function downloadResults() {
  if (!MODEL || !batchData) return;
  const headers = ['muestra', 'prediccion', 'probabilidad'];
  const rows = Array.from(document.querySelectorAll('#batch-tbody tr')).map(tr => {
    const tds = tr.querySelectorAll('td');
    return [tds[0].textContent, tds[1].textContent.trim(), tds[2].textContent];
  });
  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'predicciones_cancer.csv'; a.click();
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  await loadModels();
  buildFeatureInputs();
  setupUpload();
  document.getElementById('lr-acc-bar').style.width = '97.37%';
  document.getElementById('nn-acc-bar').style.width = '97.37%';
});
