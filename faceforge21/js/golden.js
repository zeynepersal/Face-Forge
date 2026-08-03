// js/golden.js
let goldenIntensity = 'medium';

function setGoldenIntensity(btn, level) {
  goldenIntensity = level;
  document.querySelectorAll('#goldenWarpCard .preset-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

async function runGoldenRatio() {
  if (!window.uploadedImageSrc) { alert('Upload an image first.'); return; }
  const data = await apiPost('/golden_ratio', { image: window.uploadedImageSrc });

  // Show images
  document.getElementById('goldenOrigPh').style.display = 'none';
  document.getElementById('goldenOrigImg').src = window.uploadedImageSrc;
  document.getElementById('goldenOrigImg').style.display = 'block';
  document.getElementById('goldenOverlayPh').style.display = 'none';
  document.getElementById('goldenOverlayImg').src = data.overlay;
  document.getElementById('goldenOverlayImg').style.display = 'block';

  // Overall score
  const score = data.overall;
  document.getElementById('goldenScoreCard').style.display = 'block';
  document.getElementById('goldenOverallScore').textContent = score + '%';
  setTimeout(() => {
    document.getElementById('goldenScoreBar').style.width = score + '%';
  }, 100);

  let desc = '';
  if (score >= 85)      desc = 'Exceptional — your proportions are very close to the golden ratio.';
  else if (score >= 70) desc = 'Above average — strong facial harmony with minor deviations.';
  else if (score >= 55) desc = 'Average — typical human facial proportions.';
  else                  desc = 'Below average — notable deviations from golden ratio proportions.';
  document.getElementById('goldenScoreDesc').textContent = desc;

  // Ratio breakdown table
  document.getElementById('goldenBreakdownCard').style.display = 'block';
  const tbody = document.getElementById('goldenTableBody');
  tbody.innerHTML = '';
  Object.entries(data.scores).forEach(([label, val]) => {
    const color = val.score >= 80 ? 'var(--green)' : val.score >= 60 ? 'var(--orange)' : 'var(--red)';
    tbody.innerHTML += `
      <tr>
        <td>${label}</td>
        <td class="val-normal">${val.ratio}</td>
        <td class="val-normal">${val.ideal}</td>
        <td style="color:${color};font-family:'DM Mono',monospace;font-weight:500">${val.score}%</td>
      </tr>`;
  });

  // Measurements
  document.getElementById('goldenMeasCard').style.display = 'block';
  const m = data.measurements;
  const metricsEl = document.getElementById('goldenMetrics');
  metricsEl.innerHTML = `
    <div class="metric-card m-blue"><div class="metric-label">FACE WIDTH</div><div class="metric-val" style="font-size:18px">${m.face_width}px</div></div>
    <div class="metric-card m-green"><div class="metric-label">FACE HEIGHT</div><div class="metric-val" style="font-size:18px">${m.face_height}px</div></div>
    <div class="metric-card m-purple"><div class="metric-label">NOSE WIDTH</div><div class="metric-val" style="font-size:18px">${m.nose_width}px</div></div>
    <div class="metric-card m-orange"><div class="metric-label">MOUTH WIDTH</div><div class="metric-val" style="font-size:18px">${m.mouth_width}px</div></div>
  `;

  // Side panel
  window.lastGoldenScores = data.scores;
  document.getElementById('goldenSidePanel').innerHTML = `
    <div style="margin-bottom:8px;font-size:20px;font-weight:700;color:var(--gold)">${score}%</div>
    <div style="margin-bottom:6px;color:var(--text)">${desc}</div>
    <div style="font-size:10px;color:var(--muted);margin-top:8px">φ = ${data.phi}</div>
    ${Object.entries(data.scores).map(([k,v]) => `
      <div style="display:flex;justify-content:space-between;margin-top:5px;font-size:11px">
        <span style="color:var(--muted2)">${k.split('/')[0].trim()}</span>
        <span style="color:${v.score>=75?'var(--green)':'var(--orange)'};">${v.score}%</span>
      </div>`).join('')}
  `;

  document.getElementById('ps4').className = 'step-dot step-done';
  document.getElementById('ps4').textContent = '✓';
  document.getElementById('sb-golden').textContent = '✓';
  document.getElementById('sb-golden').className = 'sidebar-badge done';

  // Reveal the golden-warp card
  document.getElementById('goldenWarpCard').style.display = 'block';
  document.getElementById('goldenWarpOrigPh').style.display = 'none';
  document.getElementById('goldenWarpOrigImg').src = window.uploadedImageSrc;
  document.getElementById('goldenWarpOrigImg').style.display = 'block';
}

async function runGoldenWarp() {
  if (!window.uploadedImageSrc) { alert('Upload an image first.'); return; }
  const btn = document.getElementById('goldenWarpBtn');
  btn.disabled = true;
  btn.textContent = 'Warping…';

  try {
    const data = await apiPost('/golden_warp', { image: window.uploadedImageSrc, intensity: goldenIntensity });

    document.getElementById('goldenWarpResultPh').style.display = 'none';
    document.getElementById('goldenWarpResultImg').src = data.result;
    document.getElementById('goldenWarpResultImg').style.display = 'block';
    window.lastTransformedSrc = data.result;

    if (data.new_overall !== null && data.new_overall !== undefined) {
      document.getElementById('goldenWarpScoreRow').style.display = 'block';
      document.getElementById('goldenWarpNewScore').textContent = data.new_overall + '%';
    }
    if (data.measured_overall !== null && data.measured_overall !== undefined) {
      const el = document.getElementById('goldenWarpMeasuredScore');
      if (el) el.textContent = 'Actually measured on the output image: ' + data.measured_overall + '%';
    }

    if (data.new_scores && Object.keys(data.new_scores).length) {
      document.getElementById('goldenOptimizedDivider').style.display = 'block';
      document.getElementById('goldenOptimizedSection').style.display = 'block';

      const prevScores = window.lastGoldenScores || {};
      const rows = Object.entries(data.new_scores).map(([k, newVal]) => {
        const label = k.split('/')[0].trim();
        const prevEntry = Object.entries(prevScores).find(([pk]) => pk.split('/')[0].trim() === label);
        const prevVal = prevEntry ? prevEntry[1].score : null;
        const delta = prevVal !== null ? Math.round((newVal - prevVal) * 10) / 10 : null;
        const deltaStr = delta === null ? '' :
          delta > 0 ? ` <span style="color:var(--green)">▲${delta}</span>` :
          delta < 0 ? ` <span style="color:var(--red)">▼${Math.abs(delta)}</span>` :
          ` <span style="color:var(--muted)">–</span>`;
        return `
          <div style="display:flex;justify-content:space-between;margin-top:5px;font-size:11px">
            <span style="color:var(--muted2)">${label}</span>
            <span style="color:${newVal>=75?'var(--green)':'var(--orange)'};">${newVal}%${deltaStr}</span>
          </div>`;
      }).join('');

      document.getElementById('goldenOptimizedPanel').innerHTML = `
        <div style="margin-bottom:8px;font-size:20px;font-weight:700;color:var(--gold)">${data.new_overall}%</div>
        <div style="margin-bottom:6px;color:var(--text)">Target score after φ-optimization.</div>
        ${rows}
      `;
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '✦ Warp to φ';
  }
}
