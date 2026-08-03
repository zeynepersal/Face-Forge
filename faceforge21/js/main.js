// js/main.js
const API = "http://localhost:5050/api";
window.uploadedImageSrc = null;

function switchNav(btn, page) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  switchPage(page);
}
function switchPage(page) {
  document.querySelectorAll('.page-section').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  if (page === 'frequency') drawSpectrums();
  if (page === 'shop') { if (window.onShopTabActivated) window.onShopTabActivated(); }
  else                  { if (window.onShopTabDeactivated) window.onShopTabDeactivated(); }
}
function setActive(item) {
  document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
  item.classList.add('active');
}

function setStatus(text, color) {
  document.getElementById('statusText').textContent = text;
  document.querySelector('.status-dot').style.background = color || 'var(--green)';
  document.querySelector('.status-dot').style.boxShadow = `0 0 6px ${color || 'var(--green)'}`;
}

async function apiPost(endpoint, body) {
  setStatus('PROCESSING', 'var(--orange)');
  try {
    const res = await fetch(API + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    setStatus('READY', 'var(--green)');
    return data;
  } catch (e) {
    setStatus('ERROR', 'var(--red)');
    alert('Backend error: ' + e.message + '\n\nMake sure backend.py is running:\npython3 backend.py');
    throw e;
  }
}

function exportPDF() {
  // Collect all available results
  const orig    = window.uploadedImageSrc;
  const aging   = document.getElementById('agingResultImg');
  const makeup  = document.getElementById('makeupResultImg');
  const golden  = document.getElementById('goldenOverallScore');
  const ageEst  = document.getElementById('ageEstValue');
  const goldenDesc = document.getElementById('goldenScoreDesc');

  const now = new Date().toLocaleString();

  let imagesHTML = '';
  if (orig) {
    imagesHTML += `<div class="img-pair">
      <div class="img-box"><div class="img-label">ORIGINAL</div><img src="${orig}"></div>`;
    if (aging && aging.src && aging.style.display !== 'none')
      imagesHTML += `<div class="img-box"><div class="img-label">AGING RESULT</div><img src="${aging.src}"></div>`;
    if (makeup && makeup.src && makeup.style.display !== 'none')
      imagesHTML += `<div class="img-box"><div class="img-label">MAKEUP RESULT</div><img src="${makeup.src}"></div>`;
    imagesHTML += '</div>';
  }

  let metricsHTML = '';
  if (golden && golden.textContent !== '—') {
    metricsHTML += `<div class="metric"><span class="metric-label">Golden Ratio Score</span><span class="metric-val">${golden.textContent}</span></div>`;
    if (goldenDesc) metricsHTML += `<div class="metric-note">${goldenDesc.textContent}</div>`;
  }
  if (ageEst && ageEst.textContent !== '—') {
    metricsHTML += `<div class="metric"><span class="metric-label">Apparent Age Estimate</span><span class="metric-val">${ageEst.textContent} yrs</span></div>`;
  }

  // Golden ratio breakdown
  let breakdownHTML = '';
  const tbody = document.getElementById('goldenTableBody');
  if (tbody && tbody.innerHTML.trim()) {
    breakdownHTML = `<h3>Golden Ratio Breakdown</h3>
    <table><thead><tr><th>Measurement</th><th>Your Ratio</th><th>Ideal</th><th>Score</th></tr></thead>
    <tbody>${tbody.innerHTML}</tbody></table>`;
  }

  const win = window.open('', '_blank');
  win.document.write(`<!DOCTYPE html><html><head><meta charset="UTF-8">
  <title>FaceForge Report</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'Inter','Segoe UI',sans-serif; color:#1a1f36; background:#fff; padding:40px; }
    h1 { font-size:26px; font-weight:800; color:#1a1f36; letter-spacing:-0.5px; }
    h2 { font-size:14px; font-weight:600; color:#4f8ef7; margin:28px 0 12px; text-transform:uppercase; letter-spacing:1px; }
    h3 { font-size:13px; font-weight:600; color:#3d4466; margin:20px 0 10px; }
    .header { display:flex; justify-content:space-between; align-items:flex-start; border-bottom:2px solid #4f8ef7; padding-bottom:16px; margin-bottom:24px; }
    .logo { font-size:22px; font-weight:800; }
    .logo span { color:#4f8ef7; }
    .date { font-size:11px; color:#8f97b4; }
    .img-pair { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px; }
    .img-box { flex:1; min-width:180px; }
    .img-label { font-size:10px; font-weight:600; color:#8f97b4; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; }
    .img-box img { width:100%; border-radius:10px; border:1px solid #dde1ee; }
    .metric { display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border:1px solid #dde1ee; border-radius:8px; margin-bottom:6px; background:#f5f6fa; }
    .metric-label { font-size:12px; color:#5a6380; }
    .metric-val { font-size:18px; font-weight:700; color:#4f8ef7; }
    .metric-note { font-size:11px; color:#8f97b4; padding:0 4px 12px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th { padding:8px 12px; text-align:left; font-size:10px; color:#8f97b4; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid #dde1ee; background:#f5f6fa; }
    td { padding:8px 12px; border-bottom:1px solid #eaecf4; color:#3d4466; }
    .footer { margin-top:40px; border-top:1px solid #dde1ee; padding-top:14px; font-size:10px; color:#b0bad4; text-align:center; }
    @media print { body { padding:20px; } }
  </style></head><body>
  <div class="header">
    <div><div class="logo">Face<span>Forge</span> <span style="font-size:11px;font-weight:400;color:#8f97b4">v2.0</span></div>
    <div style="font-size:12px;color:#5a6380;margin-top:4px">Facial Analysis Report</div></div>
    <div class="date">${now}</div>
  </div>
  ${imagesHTML ? '<h2>Images</h2>' + imagesHTML : ''}
  ${metricsHTML ? '<h2>Analysis Results</h2>' + metricsHTML : ''}
  ${breakdownHTML}
  <div class="footer">Generated by FaceForge v2.0 · Facial Transformation System</div>
  </body></html>`);
  win.document.close();
  setTimeout(() => { win.focus(); win.print(); }, 400);
}