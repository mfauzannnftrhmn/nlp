// ── State ──────────────────────────────────────────────────────────────────
const API = '';
let history = [];

// ── Particles ──────────────────────────────────────────────────────────────
function initParticles() {
  const container = document.getElementById('bgParticles');
  for (let i = 0; i < 18; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 80 + 20;
    const colors = ['#a855f740', '#3b82f640', '#06b6d430', '#22c55e30'];
    Object.assign(p.style, {
      width: size + 'px', height: size + 'px',
      left: Math.random() * 100 + '%',
      background: colors[Math.floor(Math.random() * colors.length)],
      animationDuration: (Math.random() * 20 + 15) + 's',
      animationDelay: (Math.random() * 15) + 's',
    });
    container.appendChild(p);
  }
}

// ── Tab Navigation ──────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const tab = link.dataset.tab;
      document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
      document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
      link.classList.add('active');
      document.getElementById('tab-' + tab).classList.add('active');
    });
  });
  document.getElementById('btn-try-analyzer').addEventListener('click', () => {
    document.getElementById('nav-analyzer').click();
  });
}

// ── Model Status ────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch(API + '/api/status');
    const data = await res.json();
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (data.status === 'ready') {
      dot.className = 'status-dot ok';
      txt.textContent = 'Model Ready';
      loadMetrics();
    } else {
      dot.className = 'status-dot error';
      txt.textContent = 'Model Not Found';
      showToast('⚠ Jalankan train_model.py terlebih dahulu!', 'error');
    }
  } catch {
    document.getElementById('statusDot').className = 'status-dot error';
    document.getElementById('statusText').textContent = 'Server Offline';
    showToast('❌ Server tidak dapat dihubungi. Pastikan app.py berjalan.', 'error');
  }
}

// ── Load Metrics ────────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res = await fetch(API + '/api/metrics');
    const m = await res.json();
    if (m.error) return;

    // Quick stats
    animateValue('val-accuracy', m.accuracy, '%');
    document.getElementById('val-samples').textContent = m.total_samples.toLocaleString();

    // Confusion matrix
    const [[tn, fp], [fn, tp]] = m.confusion_matrix;
    setWithAnim('val-tn', tn); setWithAnim('val-fp', fp);
    setWithAnim('val-fn', fn); setWithAnim('val-tp', tp);

    // Class metrics
    setBar('neg-precision', m.precision_negative);
    setBar('neg-recall', m.recall_negative);
    setBar('neg-f1', m.f1_negative);
    setBar('pos-precision', m.precision_positive);
    setBar('pos-recall', m.recall_positive);
    setBar('pos-f1', m.f1_positive);
    document.getElementById('support-neg').textContent = m.support_negative + ' samples';
    document.getElementById('support-pos').textContent = m.support_positive + ' samples';

    // Donut
    animateDonut(m.accuracy);
    document.getElementById('donutPct').textContent = m.accuracy + '%';
    document.getElementById('val-train').textContent = m.train_samples + ' samples';
    document.getElementById('val-test').textContent = m.test_samples + ' samples';

    showToast('✅ Metrics berhasil dimuat!', 'success');
  } catch (e) {
    console.error(e);
  }
}

function setWithAnim(id, val) {
  const el = document.getElementById(id);
  let current = 0;
  const step = val / 40;
  const timer = setInterval(() => {
    current = Math.min(current + step, val);
    el.textContent = Math.round(current);
    if (current >= val) clearInterval(timer);
  }, 30);
}

function setBar(key, val) {
  setTimeout(() => {
    document.getElementById('bar-' + key).style.width = val + '%';
    document.getElementById('text-' + key).textContent = val + '%';
  }, 400);
}

function animateValue(id, val, suffix = '') {
  const el = document.getElementById(id);
  let cur = 0;
  const step = val / 50;
  const t = setInterval(() => {
    cur = Math.min(cur + step, val);
    el.textContent = cur.toFixed(1) + suffix;
    if (cur >= val) { el.textContent = val + suffix; clearInterval(t); }
  }, 25);
}

function animateDonut(pct) {
  const circumference = 2 * Math.PI * 80; // r=80
  setTimeout(() => {
    const track = document.getElementById('donutTrack');
    const dash = (pct / 100) * circumference;
    track.style.strokeDasharray = `${dash} ${circumference}`;
  }, 300);
}

// ── Analyzer ────────────────────────────────────────────────────────────────
function initAnalyzer() {
  const textarea = document.getElementById('textInput');
  const charCount = document.getElementById('charCount');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnClear = document.getElementById('btn-clear');
  const btnAgain = document.getElementById('btn-analyze-again');

  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length + ' / 500';
  });

  btnClear.addEventListener('click', () => {
    textarea.value = '';
    charCount.textContent = '0 / 500';
    showIdle();
  });

  btnAnalyze.addEventListener('click', () => analyzeSentiment());
  btnAgain.addEventListener('click', () => { showIdle(); textarea.focus(); });

  // Sample buttons
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.dataset.text;
      charCount.textContent = btn.dataset.text.length + ' / 500';
      analyzeSentiment();
    });
  });

  // Clear history
  document.getElementById('btn-clear-history').addEventListener('click', () => {
    history = [];
    renderHistory();
  });
}

async function analyzeSentiment() {
  const textarea = document.getElementById('textInput');
  const text = textarea.value.trim();
  if (!text) { showToast('⚠ Masukkan teks terlebih dahulu!', 'error'); return; }

  setAnalyzeLoading(true);

  try {
    const res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showResult(data);
    addHistory(data);
  } catch (e) {
    showToast('❌ ' + (e.message || 'Gagal terhubung ke server'), 'error');
  } finally {
    setAnalyzeLoading(false);
  }
}

function setAnalyzeLoading(loading) {
  const btn = document.getElementById('btn-analyze');
  btn.querySelector('.btn-text').style.display = loading ? 'none' : '';
  btn.querySelector('.btn-loading').style.display = loading ? 'flex' : 'none';
  btn.disabled = loading;
}

function showIdle() {
  document.getElementById('resultIdle').style.display = '';
  document.getElementById('resultDisplay').style.display = 'none';
  const panel = document.querySelector('.analyzer-result-panel');
  panel.style.alignItems = 'center';
  panel.style.justifyContent = 'center';
}

function showResult(data) {
  document.getElementById('resultIdle').style.display = 'none';
  const display = document.getElementById('resultDisplay');
  display.style.display = 'block';
  const panel = document.querySelector('.analyzer-result-panel');
  panel.style.alignItems = 'flex-start';
  panel.style.justifyContent = 'flex-start';

  const isPos = data.sentiment === 'positive';
  const badge = document.getElementById('resultBadge');
  badge.className = 'result-sentiment-badge ' + (isPos ? 'positive' : 'negative');
  document.getElementById('sentimentEmoji').textContent = isPos ? '😊' : '😞';
  document.getElementById('sentimentLabel').textContent = isPos ? 'POSITIF' : 'NEGATIF';
  document.getElementById('sentimentLabel').style.color = isPos ? '#22c55e' : '#ef4444';
  document.getElementById('sentimentSublabel').textContent =
    isPos ? 'Ulasan ini mengandung sentimen positif' : 'Ulasan ini mengandung sentimen negatif';

  // Confidence
  document.getElementById('confidencePct').textContent = data.confidence + '%';
  setTimeout(() => {
    document.getElementById('confidenceFill').style.width = data.confidence + '%';
  }, 100);

  // Probabilities
  document.getElementById('prob-neg').textContent = data.probabilities.negative + '%';
  document.getElementById('prob-pos').textContent = data.probabilities.positive + '%';
  setTimeout(() => {
    document.getElementById('bar-prob-neg').style.width = data.probabilities.negative + '%';
    document.getElementById('bar-prob-pos').style.width = data.probabilities.positive + '%';
  }, 200);

  // Text preview
  document.getElementById('analyzedText').textContent = data.text;
}

// ── History ──────────────────────────────────────────────────────────────────
function addHistory(data) {
  history.unshift(data);
  if (history.length > 12) history.pop();
  renderHistory();
}

function renderHistory() {
  const section = document.getElementById('historySection');
  const grid = document.getElementById('historyGrid');
  if (history.length === 0) { section.style.display = 'none'; return; }
  section.style.display = '';
  grid.innerHTML = history.map(item => {
    const isPos = item.sentiment === 'positive';
    const color = isPos ? '#22c55e' : '#ef4444';
    return `
      <div class="history-item">
        <div class="hist-emoji">${isPos ? '😊' : '😞'}</div>
        <div style="flex:1;overflow:hidden">
          <div class="hist-text">${escapeHtml(item.text)}</div>
          <div class="hist-label" style="color:${color}">${isPos ? 'POSITIF' : 'NEGATIF'}</div>
          <div class="hist-conf">${item.confidence}% confidence</div>
        </div>
      </div>`;
  }).join('');
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Toast ──────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = 'toast ' + type + ' show';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Donut SVG Gradient ─────────────────────────────────────────────────────
function addDonutGradient() {
  // Minimal style: use solid color via CSS (.donut-track stroke set in CSS)
  const track = document.getElementById('donutTrack');
  if (track) track.style.stroke = '#18181b';
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initTabs();
  addDonutGradient();
  initAnalyzer();
  checkStatus();
});
