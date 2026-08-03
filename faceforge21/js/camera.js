// js/camera.js
let cameraStream  = null;
let facingMode    = 'user';
let arInterval    = null;
let arRunning     = false;

async function openCamera() {
  const modal = document.getElementById('cameraModal');
  modal.style.display = 'flex';
  await startStream();
}

async function startStream() {
  try {
    if (cameraStream) cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    document.getElementById('cameraFeed').srcObject = cameraStream;
  } catch (err) {
    closeCamera();
    alert('Camera access denied or not available.\n\n' + err.message);
  }
}

function closeCamera() {
  stopLiveAR();
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  document.getElementById('cameraFeed').srcObject = null;
  document.getElementById('cameraModal').style.display = 'none';
  const _arToggle = document.getElementById('arLiveToggle');
  if (_arToggle) _arToggle.checked = false;
  const _arImg = document.getElementById('arPreviewImg');
  if (_arImg) _arImg.style.display = 'none';
  const _arPh = document.getElementById('arPreviewPh');
  if (_arPh) _arPh.style.display = 'flex';
  const _arSt = document.getElementById('arStatus');
  if (_arSt) _arSt.textContent = 'INACTIVE';
}

async function flipCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  await startStream();
}

function grabFrame() {
  const video  = document.getElementById('cameraFeed');
  const canvas = document.getElementById('cameraCanvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  if (facingMode === 'user') {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
}

function getMakeupBody(imageDataUrl) {
  function val(id)    { const el = document.getElementById(id); return el ? parseInt(el.value) : 0; }
  function checked(id){ const el = document.getElementById(id); return el ? el.checked : false; }
  function hex(id) {
    const el = document.getElementById(id);
    if (!el) return [128,128,128];
    const h = el.value;
    return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
  }
  return {
    image:            imageDataUrl,
    lip_color:        hex('pick-lipcolor'),
    lip_opacity:      val('sl-lipopacity'),
    lip_gloss:        checked('tog-lipgloss')   ? val('sl-lipgloss')   : 0,
    eyeliner:         checked('tog-eyeliner')   ? val('sl-eyeliner')   : 0,
    blush:            checked('tog-blush')      ? val('sl-blush')      : 0,
    blush_color:      hex('pick-blush'),
    foundation:       checked('tog-foundation') ? val('sl-foundation') : 0,
    foundation_color: hex('pick-foundation'),
    highlight:        checked('tog-highlight')  ? val('sl-highlight')  : 0,
  };
}

function toggleLiveAR(enabled) {
  if (enabled) startLiveAR();
  else         stopLiveAR();
}

function startLiveAR() {
  if (arRunning) return;
  arRunning = true;
  setARStatus('PROCESSING...', 'var(--orange)');
  sendARFrame();
  arInterval = setInterval(sendARFrame, 2200);
}

function stopLiveAR() {
  arRunning = false;
  if (arInterval) { clearInterval(arInterval); arInterval = null; }
  setARStatus('INACTIVE', 'var(--muted)');
}

function setARStatus(text, color) {
  const el = document.getElementById('arStatus');
  if (el) { el.textContent = text; el.style.color = color; }
}

async function sendARFrame() {
  if (!arRunning || !cameraStream) return;
  const frameDataUrl = grabFrame();
  const body = getMakeupBody(frameDataUrl);

  const anyActive = body.lip_opacity > 0 || body.lip_gloss > 0 || body.eyeliner > 0 ||
                    body.blush > 0 || body.foundation > 0 || body.highlight > 0;

  if (!anyActive) {
    showARResult(frameDataUrl);
    setARStatus('NO MAKEUP ACTIVE', 'var(--muted2)');
    return;
  }

  setARStatus('PROCESSING...', 'var(--orange)');
  try {
    const res  = await fetch('http://localhost:5050/api/makeup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.result) {
      showARResult(data.result);
      setARStatus('LIVE ●', 'var(--green)');
    }
  } catch (e) {
    setARStatus('ERROR — backend running?', 'var(--red)');
  }
}

function showARResult(src) {
  document.getElementById('arPreviewImg').src = src;
  document.getElementById('arPreviewImg').style.display = 'block';
  document.getElementById('arPreviewPh').style.display  = 'none';
}

function capturePhoto() {
  const dataUrl = grabFrame();
  window.uploadedImageSrc = dataUrl;

  const zone = document.getElementById('uploadZone');
  zone.classList.add('has-image');
  zone.innerHTML = `<img src="${dataUrl}" style="max-height:160px;border-radius:8px;object-fit:cover"><div style="font-size:12px;color:var(--green);margin-top:8px">✓ Photo captured — click to replace</div><input type="file" id="imageInput" accept="image/*" style="display:none" onchange="handleImageUpload(event)">`;

  document.getElementById('previewRow').style.display = 'grid';
  document.getElementById('originalImg').src  = dataUrl;
  document.getElementById('processedImg').src = dataUrl;

  ['warpOrigImg','agingOrigImg','goldenOrigImg','makeupOrigImg'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.src = dataUrl; el.style.display = 'block'; }
  });
  ['warpOrigPh','agingOrigPh','goldenOrigPh','makeupOrigPh'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });

  const canvas = document.getElementById('cameraCanvas');
  document.getElementById('infName').textContent   = 'camera.png';
  document.getElementById('infFormat').textContent = 'PNG';
  document.getElementById('infSize').textContent   = '—';
  document.getElementById('infDims').textContent   = canvas.width + '×' + canvas.height;

  document.getElementById('ps1').className   = 'step-dot step-done';
  document.getElementById('ps1').textContent = '✓';
  document.getElementById('sb-upload').textContent = '✓';
  document.getElementById('sb-upload').className   = 'sidebar-badge done';

  closeCamera();
}

document.addEventListener('click', e => {
  const modal = document.getElementById('cameraModal');
  if (e.target === modal) closeCamera();
});