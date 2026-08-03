// js/frequency.js

function fmt(n) {
  if (n >= 1e12) return (n/1e12).toFixed(2) + 'T';
  if (n >= 1e9)  return (n/1e9).toFixed(2)  + 'B';
  if (n >= 1e6)  return (n/1e6).toFixed(2)  + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toFixed(0);
}

async function drawSpectrums() {
  if (!window.uploadedImageSrc) {
    drawSimSpectrum('specOrig', false);
    drawSimSpectrum('specTransform', true);
    return;
  }

  try {
    // Fetch both original and transformed (or same if no transform yet)
    const src1 = window.uploadedImageSrc;
    const src2 = window.lastTransformedSrc || window.uploadedImageSrc;

    const [res1, res2] = await Promise.all([
      fetch('http://localhost:5050/api/frequency', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({image: src1})
      }),
      fetch('http://localhost:5050/api/frequency', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({image: src2})
      })
    ]);

    const d1 = await res1.json();
    const d2 = await res2.json();

    // Draw spectrums on canvases
    drawImgOnCanvas('specOrig',      d1.spectrum, 'ORIGINAL',    'rgba(79,142,247,0.85)');
    drawImgOnCanvas('specTransform', d2.spectrum, 'TRANSFORMED', 'rgba(31,232,159,0.85)');

    // Update metric cards
    const s1 = d1.stats, s2 = d2.stats;
    const totalOrig  = d1.total_energy  || 0;
    const totalTrans = d2.total_energy  || 0;
    const hflfOrig   = d1.hflf_ratio   || 0;
    const hflfTrans  = d2.hflf_ratio   || 0;

    setEl('freqEnergyOrig',  fmt(totalOrig));
    setEl('freqEnergyTrans', fmt(totalTrans));
    setEl('freqHFLFOrig',    hflfOrig.toFixed(3));
    setEl('freqHFLFTrans',   hflfTrans.toFixed(3));

    const hflfDelta = totalOrig > 0 ? ((hflfTrans - hflfOrig) / (hflfOrig + 1e-9) * 100) : 0;
    setEl('freqHFLFDelta', (hflfDelta >= 0 ? '+' : '') + hflfDelta.toFixed(1) + '% vs original');

    // Update table
    const lfOrig  = d1.lf_energy  || 0;
    const lfTrans = d2.lf_energy  || 0;
    const hfOrig  = d1.hf_energy  || 0;
    const hfTrans = d2.hf_energy  || 0;

    const delta = (a, b) => {
      if (a === 0) return '—';
      const d = (b - a) / (a + 1e-9) * 100;
      const cls = d > 0 ? 'val-mid' : 'val-good';
      return `<span class="${cls}">${d >= 0 ? '+' : ''}${d.toFixed(2)}%</span>`;
    };

    const tbody = document.getElementById('freqTableBody');
    if (tbody) {
      tbody.innerHTML = `
        <tr><td>Total Spectral Energy</td><td class="val-normal">${fmt(totalOrig)}</td><td class="val-normal">${fmt(totalTrans)}</td><td>${delta(totalOrig, totalTrans)}</td></tr>
        <tr><td>Low Frequency Energy</td><td class="val-normal">${fmt(lfOrig)}</td><td class="val-normal">${fmt(lfTrans)}</td><td>${delta(lfOrig, lfTrans)}</td></tr>
        <tr><td>High Frequency Energy</td><td class="val-normal">${fmt(hfOrig)}</td><td class="val-normal">${fmt(hfTrans)}</td><td>${delta(hfOrig, hfTrans)}</td></tr>
        <tr><td>HF / LF Ratio</td><td class="val-normal">${hflfOrig.toFixed(3)}</td><td class="val-normal">${hflfTrans.toFixed(3)}</td><td>${delta(hflfOrig, hflfTrans)}</td></tr>
      `;
    }

  } catch (e) {
    console.error('Frequency API error:', e);
    drawSimSpectrum('specOrig', false);
    drawSimSpectrum('specTransform', true);
  }
}

function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function drawImgOnCanvas(canvasId, src, label, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.fillStyle = color;
    ctx.font = '10px DM Mono, monospace';
    ctx.fillText(label, 8, 18);
  };
  img.src = src;
}

function drawSimSpectrum(id, transformed) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#08090d';
  ctx.fillRect(0, 0, w, h);
  const cx = w/2, cy = h/2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx=x-cx, dy=y-cy;
      const r = Math.sqrt(dx*dx+dy*dy);
      const angle = Math.atan2(dy, dx);
      let power = 1500/(1+r*0.6) + 80/(1+r*0.1) + 30*Math.abs(Math.sin(angle*4))/(1+r*0.2);
      if (transformed && r > 20) power += r * 0.8;
      power = Math.min(255, Math.log1p(power)*22);
      const t = power/255;
      ctx.fillStyle = `rgb(${Math.floor(t*180)},${Math.floor(t*t*220)},${Math.floor(power*0.9)})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  ctx.strokeStyle='rgba(255,255,255,0.06)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(w,cy); ctx.stroke();
  ctx.fillStyle = transformed ? 'rgba(31,232,159,0.7)' : 'rgba(79,142,247,0.7)';
  ctx.font = '10px DM Mono, monospace';
  ctx.fillText(transformed ? 'TRANSFORMED' : 'ORIGINAL', 8, 18);
}

window.addEventListener('load', drawSpectrums);