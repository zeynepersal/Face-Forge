// js/aging.js
let agingMode = 'aging';

function updateAgingMode(mode) {
  agingMode = mode;
  document.getElementById('agingModeLabel').textContent = mode === 'aging' ? 'AGING' : 'DE-AGING';
  document.getElementById('agingDesc').textContent = mode === 'aging'
    ? 'Boosts high-frequency components to simulate wrinkles'
    : 'Applies low-pass filter to smooth skin, preserving edges';
}

function selectPreset(btn, mode, intensity) {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  agingMode = mode === 'custom' ? agingMode : mode;
  document.getElementById('sl-aging').value = intensity;
  document.getElementById('sv-aging').textContent = intensity + '%';
  if (mode !== 'custom') {
    document.querySelectorAll('input[name="agingMode"]').forEach(r => r.checked = r.value === mode);
    updateAgingMode(mode);
  }
  const labels = { aging: 'AGED', deaging: 'DE-AGED', custom: 'CUSTOM' };
  document.getElementById('agingResultLabel').textContent = labels[mode] || 'RESULT';
}

async function runAging(options = {}) {
  if (!window.uploadedImageSrc) { alert('Upload an image first.'); return; }
  const promptOnly = !!options.promptOnly;

  const ph  = document.getElementById('agingResultPh');
  const img = document.getElementById('agingResultImg');
  if (ph)  { ph.innerHTML  = '<span style="color:var(--muted2);font-size:12px">✨ Generating with AI…</span>'; ph.style.display = 'flex'; }
  if (img) img.style.display = 'none';

  const data = await apiPost('/aging', {
    image:     window.uploadedImageSrc,
    mode:      agingMode,
    intensity: document.getElementById('sl-aging').value,
    prompt:    promptOnly ? (document.getElementById('agingPrompt')?.value.trim() || '') : '',
    promptOnly,
  });

  if (ph)  ph.style.display = 'none';
  if (img) { img.src = data.result; img.style.display = 'block'; }
  window.lastTransformedSrc = data.result;
  document.getElementById('sb-aging').textContent = '✓';
  document.getElementById('sb-aging').className = 'sidebar-badge done';
}

function applyAgingPrompt() {
  const prompt = document.getElementById('agingPrompt')?.value.trim() || '';
  if (!prompt) { alert('Write a prompt first.'); return; }
  runAging({ promptOnly: true });
}
