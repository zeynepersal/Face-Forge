// js/evaluation.js

async function runEvaluation() {
  if (!window.uploadedImageSrc) {
    alert('Upload an image first.');
    return;
  }
  if (!window.lastTransformedSrc) {
    alert('Apply a transformation first (Expression, Aging, or Makeup), then evaluate.');
    return;
  }

  const btn = document.getElementById('evalBtn');
  if (btn) { btn.textContent = 'PROCESSING...'; btn.disabled = true; }

  try {
    const res = await fetch('http://localhost:5050/api/evaluation', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        original:    window.uploadedImageSrc,
        transformed: window.lastTransformedSrc
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

    setEl('evalMSE',    data.mse);
    setEl('evalPSNR',   data.psnr + ' dB');
    setEl('evalSSIM',   data.ssim);

    const statusColor = data.status === 'GOOD' ? 'var(--green)' : data.status === 'MED' ? 'var(--orange)' : 'var(--red)';
    const statusEl = document.getElementById('evalStatus');
    if (statusEl) { statusEl.textContent = data.status; statusEl.style.color = statusColor; }

    // Progress bars
    const mseBar  = document.getElementById('evalMSEBar');
    const psnrBar = document.getElementById('evalPSNRBar');
    const ssimBar = document.getElementById('evalSSIMBar');
    if (mseBar)  mseBar.style.width  = Math.min(100, data.mse / 300 * 100).toFixed(0) + '%';
    if (psnrBar) psnrBar.style.width = Math.min(100, (data.psnr - 20) / 20 * 100).toFixed(0) + '%';
    if (ssimBar) ssimBar.style.width = (data.ssim * 100).toFixed(0) + '%';

    // Results table
    const tbody = document.getElementById('evalTableBody');
    if (tbody) {
      const color = data.status === 'GOOD' ? 'var(--green)' : data.status === 'MED' ? 'var(--orange)' : 'var(--red)';
      const mseClass  = data.mse  < 100 ? 'val-good' : data.mse  < 200 ? 'val-normal' : 'val-mid';
      const psnrClass = data.psnr > 30  ? 'val-good' : data.psnr > 25  ? 'val-normal' : 'val-mid';
      const ssimClass = data.ssim > 0.9 ? 'val-good' : data.ssim > 0.8 ? 'val-normal' : 'val-mid';
      tbody.innerHTML = `
        <tr>
          <td>Last Transformation</td>
          <td class="${mseClass}">${data.mse}</td>
          <td class="${psnrClass}">${data.psnr}</td>
          <td class="${ssimClass}">${data.ssim}</td>
          <td><span style="color:${color};font-size:11px">● ${data.status}</span></td>
        </tr>`;
    }

    // Notification
    const notif = document.getElementById('evalNotif');
    if (notif) {
      if (data.ssim > 0.79) {
        notif.className = 'notification notif-success';
        notif.textContent = '✓ SSIM > 0.79 — perceptual quality maintained.';
      } else {
        notif.className = 'notification notif-warn';
        notif.textContent = '⚠ SSIM below 0.79 — significant perceptual difference detected.';
      }
    }

    if (btn) { btn.textContent = '✓ DONE'; btn.disabled = false; btn.style.color = 'var(--green)'; }

  } catch (e) {
    alert('Evaluation error: ' + e.message + '\n\nMake sure backend.py is running.');
    if (btn) { btn.textContent = '▶ Evaluate'; btn.disabled = false; }
  }
}