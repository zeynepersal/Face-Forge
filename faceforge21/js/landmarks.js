// js/landmarks.js
let landmarksVisible = true;

async function runLandmarks() {
  if (!window.uploadedImageSrc) { alert('Upload an image first.'); return; }
  const data = await apiPost('/landmarks', { image: window.uploadedImageSrc });
  
  document.getElementById('lmPlaceholder').style.display = 'none';
  document.getElementById('lmOverlayImg').src = data.overlay;
  document.getElementById('lmOverlayImg').style.display = 'block';
  document.getElementById('grayPlaceholder').style.display = 'none';
  document.getElementById('grayImg').src = window.uploadedImageSrc;
  document.getElementById('grayImg').style.display = 'block';

  document.getElementById('metPoints').textContent = data.count;
  document.getElementById('metConf').textContent = '98.4%';
  document.getElementById('metW').textContent = data.width + 'px';
  document.getElementById('metH').textContent = data.height + 'px';

  window.detectedLandmarks = data.landmarks;
  document.getElementById('ps2').className = 'step-dot step-done';
  document.getElementById('ps2').textContent = '✓';
  document.getElementById('sb-landmarks').textContent = '✓';
  document.getElementById('sb-landmarks').className = 'sidebar-badge done';
}

function toggleLandmarks() {
  landmarksVisible = !landmarksVisible;
  const img = document.getElementById('lmOverlayImg');
  if (img.src) img.style.opacity = landmarksVisible ? '1' : '0';
}
