// js/makeup.js

// ── Live Makeup Preview (instant canvas-based) ────────────────────────────────
let makeupLiveStream     = null;
let makeupLiveLandmarker = null;
let makeupLiveRunning    = false;

async function startMakeupLive() {
  const card = document.getElementById('makeupLiveCard');
  if (card) card.style.display = 'block';

  const video = document.getElementById('makeupLiveVideo');
  try {
    if (makeupLiveStream) makeupLiveStream.getTracks().forEach(t => t.stop());
    makeupLiveStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = makeupLiveStream;
    video.addEventListener('loadeddata', () => {
      const canvas = document.getElementById('makeupLiveCanvas');
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      if (!makeupLiveRunning) {
        makeupLiveRunning = true;
        if (!makeupLiveLandmarker) initMakeupLandmarker();
        makeupLiveLoop();
      }
    }, { once: true });
  } catch (e) {
    alert('Camera access denied: ' + e.message);
  }
}

function stopMakeupLive() {
  makeupLiveRunning = false;
  if (makeupLiveStream) { makeupLiveStream.getTracks().forEach(t => t.stop()); makeupLiveStream = null; }
  const video = document.getElementById('makeupLiveVideo');
  if (video) video.srcObject = null;
  const card = document.getElementById('makeupLiveCard');
  if (card) card.style.display = 'none';
}

async function initMakeupLandmarker() {
  try {
    const { FaceLandmarker, FilesetResolver } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );
    const resolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    makeupLiveLandmarker = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: 'http://localhost:5050/api/model/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  } catch (e) {
    console.warn('Makeup live landmarker failed:', e.message);
  }
}

function makeupLiveLoop() {
  if (!makeupLiveRunning) return;
  const video  = document.getElementById('makeupLiveVideo');
  const canvas = document.getElementById('makeupLiveCanvas');
  if (!video || !canvas) return;
  const ctx = canvas.getContext('2d');

  if (video.readyState >= 2) {
    // Draw mirrored camera feed
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    // Overlay makeup
    if (makeupLiveLandmarker) {
      try {
        const res = makeupLiveLandmarker.detectForVideo(video, performance.now());
        if (res.faceLandmarks?.length > 0) {
          drawMakeupCanvas(ctx, res.faceLandmarks[0], canvas.width, canvas.height);
        }
      } catch (_) {}
    }
  }
  requestAnimationFrame(makeupLiveLoop);
}

// Convert hex color + alpha to rgba() string
function _hexRgba(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

// Draw all enabled makeup layers onto ctx using face landmarks (real-time, no backend)
function drawMakeupCanvas(ctx, landmarks, cw, ch) {
  // Landmark → canvas pixel; mirrors x to match selfie video flip
  const pt = idx => {
    const p = landmarks[idx];
    return p ? { x: (1 - p.x) * cw, y: p.y * ch } : null;
  };

  const v  = id => parseInt(document.getElementById(id)?.value || 0);
  const on = id => !!document.getElementById(id)?.checked;

  const leftEdge  = pt(234);
  const rightEdge = pt(454);
  const fw = leftEdge && rightEdge ? Math.abs(rightEdge.x - leftEdge.x) : cw * 0.3;

  // ── Foundation (face oval, subtle skin-tone wash) ────────────────────────
  if (on('tog-foundation') && v('sl-foundation') > 0) {
    const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                       152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
    const pts = FACE_OVAL.map(i => pt(i)).filter(Boolean);
    if (pts.length >= 10) {
      ctx.save();
      ctx.globalAlpha = v('sl-foundation') / 100 * 0.28;
      ctx.fillStyle   = document.getElementById('pick-foundation').value;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  // ── Lip color ────────────────────────────────────────────────────────────
  const lipAlpha = v('sl-lipopacity') / 100 * 0.78;
  if (lipAlpha > 0) {
    const lipColor = document.getElementById('pick-lipcolor').value;
    const UPPER = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78];
    const LOWER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78];
    for (const ring of [UPPER, LOWER]) {
      const pts = ring.map(i => pt(i)).filter(Boolean);
      if (pts.length < 3) continue;
      ctx.save();
      ctx.globalAlpha = lipAlpha;
      ctx.fillStyle   = lipColor;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  // ── Lip gloss (specular center highlight) ────────────────────────────────
  if (on('tog-lipgloss') && v('sl-lipgloss') > 0) {
    const glossStr = v('sl-lipgloss') / 100;
    const center   = pt(13);
    if (center) {
      const gw = fw * 0.09;
      const grad = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, gw);
      grad.addColorStop(0,   `rgba(255,255,255,${glossStr * 0.75})`);
      grad.addColorStop(0.4, `rgba(255,255,255,${glossStr * 0.35})`);
      grad.addColorStop(1,   'rgba(255,255,255,0)');
      ctx.save();
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(center.x, center.y, gw, gw * 0.45, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // ── Eyeliner (upper lash lines) ──────────────────────────────────────────
  if (on('tog-eyeliner') && v('sl-eyeliner') > 0) {
    const alpha = v('sl-eyeliner') / 100 * 0.85;
    const lw    = Math.max(1.5, fw * 0.007);
    const L_LASH = [33,246,161,160,159,158,157,173,133];
    const R_LASH = [362,398,384,385,386,387,388,466,263];
    for (const lashSet of [L_LASH, R_LASH]) {
      const pts = lashSet.map(i => pt(i)).filter(Boolean);
      if (pts.length < 2) continue;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = '#111111';
      ctx.lineWidth   = lw;
      ctx.lineCap     = 'round';
      ctx.lineJoin    = 'round';
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
      ctx.restore();
    }
  }

  // ── Blush (radial gradient on cheeks) ───────────────────────────────────
  if (on('tog-blush') && v('sl-blush') > 0) {
    const blushStr   = v('sl-blush') / 100;
    const blushColor = document.getElementById('pick-blush').value;
    const br         = fw * 0.18;
    for (const cheekIdx of [205, 425]) {
      const c = pt(cheekIdx);
      if (!c) continue;
      const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, br);
      grad.addColorStop(0,   _hexRgba(blushColor, blushStr * 0.28));
      grad.addColorStop(0.5, _hexRgba(blushColor, blushStr * 0.12));
      grad.addColorStop(1,   _hexRgba(blushColor, 0));
      ctx.save();
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, br, br * 0.62, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // ── Highlighter (nose bridge + cheekbones) ───────────────────────────────
  if (on('tog-highlight') && v('sl-highlight') > 0) {
    const hlStr = v('sl-highlight') / 100;
    const hlR   = fw * 0.08;
    for (const hlIdx of [168, 116, 345]) {
      const c = pt(hlIdx);
      if (!c) continue;
      const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, hlR);
      grad.addColorStop(0, `rgba(255,255,255,${hlStr * 0.38})`);
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.save();
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, hlR, hlR * 0.8, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }
}

const MAKEUP_PRESETS = {
  natural: {
    lipcolor: '#c87e6a', lipopacity: 35, lipgloss: false, glossval: 0,
    eyeliner: false, linerval: 0,
    blush: true, blushval: 30, blushcolor: '#e8a090',
    foundation: true, foundval: 40, foundcolor: '#f5c5a3',
    highlight: true, hlval: 25,
  },
  glam: {
    lipcolor: '#8b0033', lipopacity: 85, lipgloss: true, glossval: 60,
    eyeliner: true, linerval: 80,
    blush: true, blushval: 55, blushcolor: '#d4607a',
    foundation: true, foundval: 65, foundcolor: '#f0b090',
    highlight: true, hlval: 70,
  },
  smoky: {
    lipcolor: '#3a1a1a', lipopacity: 70, lipgloss: false, glossval: 0,
    eyeliner: true, linerval: 90,
    blush: false, blushval: 0, blushcolor: '#c06070',
    foundation: true, foundval: 50, foundcolor: '#e8b898',
    highlight: false, hlval: 0,
  },
  coral: {
    lipcolor: '#e8603a', lipopacity: 75, lipgloss: true, glossval: 40,
    eyeliner: false, linerval: 0,
    blush: true, blushval: 50, blushcolor: '#f0845a',
    foundation: true, foundval: 45, foundcolor: '#f5c5a3',
    highlight: true, hlval: 40,
  },
  editorial: {
    lipcolor: '#1a0a2e', lipopacity: 90, lipgloss: false, glossval: 0,
    eyeliner: true, linerval: 95,
    blush: false, blushval: 0, blushcolor: '#c060a0',
    foundation: true, foundval: 70, foundcolor: '#f0c0a0',
    highlight: true, hlval: 80,
  },
};

function applyMakeupPreset(name) {
  const p = MAKEUP_PRESETS[name];
  if (!p) return;

  // Mark active preset
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');

  // Lip
  document.getElementById('pick-lipcolor').value        = p.lipcolor;
  document.getElementById('sl-lipopacity').value        = p.lipopacity;
  document.getElementById('sv-lipopacity').textContent  = p.lipopacity + '%';

  document.getElementById('tog-lipgloss').checked       = p.lipgloss;
  document.getElementById('sl-lipgloss').disabled       = !p.lipgloss;
  document.getElementById('sl-lipgloss').value          = p.glossval;
  document.getElementById('sv-lipgloss').textContent    = p.glossval + '%';

  // Eyeliner
  document.getElementById('tog-eyeliner').checked       = p.eyeliner;
  document.getElementById('sl-eyeliner').disabled       = !p.eyeliner;
  document.getElementById('sl-eyeliner').value          = p.linerval;
  document.getElementById('sv-eyeliner').textContent    = p.linerval + '%';

  // Blush
  document.getElementById('tog-blush').checked          = p.blush;
  document.getElementById('sl-blush').disabled          = !p.blush;
  document.getElementById('sl-blush').value             = p.blushval;
  document.getElementById('sv-blush').textContent       = p.blushval + '%';
  document.getElementById('pick-blush').value           = p.blushcolor;

  // Foundation
  document.getElementById('tog-foundation').checked     = p.foundation;
  document.getElementById('sl-foundation').disabled     = !p.foundation;
  document.getElementById('sl-foundation').value        = p.foundval;
  document.getElementById('sv-foundation').textContent  = p.foundval + '%';
  document.getElementById('pick-foundation').value      = p.foundcolor;

  // Highlighter
  document.getElementById('tog-highlight').checked      = p.highlight;
  document.getElementById('sl-highlight').disabled      = !p.highlight;
  document.getElementById('sl-highlight').value         = p.hlval;
  document.getElementById('sv-highlight').textContent   = p.hlval + '%';
}


function toggleMakeup(name) {
  const tog = document.getElementById('tog-' + name);
  const sl  = document.getElementById('sl-'  + name);
  if (!tog || !sl) return;
  sl.disabled = !tog.checked;
  if (tog.checked && sl.value === '0') sl.value = 50;
  document.getElementById('sv-' + name).textContent = sl.value + '%';
}

// Live slider label update for lip opacity (always enabled)
document.addEventListener('DOMContentLoaded', () => {
  const lipOp = document.getElementById('sl-lipopacity');
  if (lipOp) {
    lipOp.addEventListener('input', () => {
      document.getElementById('sv-lipopacity').textContent = lipOp.value + '%';
    });
  }

  // Propagate uploaded image to makeup panel
  const origObserver = new MutationObserver(() => {
    const src = window.uploadedImageSrc;
    if (!src) return;
    const el = document.getElementById('makeupOrigImg');
    if (el && !el.src) {
      el.src = src;
      el.style.display = 'block';
      document.getElementById('makeupOrigPh').style.display = 'none';
    }
  });
  // Poll for image upload (simple approach compatible with existing upload.js)
  setInterval(() => {
    if (window.uploadedImageSrc) {
      const el = document.getElementById('makeupOrigImg');
      if (el && el.src !== window.uploadedImageSrc) {
        el.src = window.uploadedImageSrc;
        el.style.display = 'block';
        const ph = document.getElementById('makeupOrigPh');
        if (ph) ph.style.display = 'none';
      }
    }
  }, 500);
});

function resetMakeup() {
  ['eyeliner','blush','foundation','highlight','lipgloss','haircolor'].forEach(n => {
    const tog = document.getElementById('tog-' + n);
    const sl  = document.getElementById('sl-'  + n);
    if (tog) { tog.checked = false; }
    if (sl)  { sl.value = 0; sl.disabled = true; }
    const sv = document.getElementById('sv-' + n);
    if (sv) sv.textContent = '0%';
  });
  document.getElementById('sl-lipopacity').value = 60;
  document.getElementById('sv-lipopacity').textContent = '60%';
  document.getElementById('pick-foundation').value = '#f5c5a3';
  document.getElementById('pick-blush').value      = '#e06080';

  document.getElementById('makeupResultImg').style.display = 'none';
  document.getElementById('makeupResultPh').style.display = 'flex';
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return [r, g, b];
}

async function runMakeup() {
  if (!window.uploadedImageSrc) { alert('Upload an image first.'); return; }

  const body = {
    image: window.uploadedImageSrc,

    // Lips
    lip_color:   hexToRgb(document.getElementById('pick-lipcolor').value),
    lip_opacity: parseInt(document.getElementById('sl-lipopacity').value),
    lip_gloss:   document.getElementById('tog-lipgloss').checked
                   ? parseInt(document.getElementById('sl-lipgloss').value) : 0,

    // Eyes
    eyeliner:    document.getElementById('tog-eyeliner').checked
                   ? parseInt(document.getElementById('sl-eyeliner').value) : 0,
    blush:       document.getElementById('tog-blush').checked
                   ? parseInt(document.getElementById('sl-blush').value) : 0,
    blush_color: hexToRgb(document.getElementById('pick-blush').value),

    // Skin
    foundation:  document.getElementById('tog-foundation').checked
                   ? parseInt(document.getElementById('sl-foundation').value) : 0,
    foundation_color: hexToRgb(document.getElementById('pick-foundation').value),
    highlight:   document.getElementById('tog-highlight').checked
                   ? parseInt(document.getElementById('sl-highlight').value) : 0,
  };

  const data = await apiPost('/makeup', body);

  let resultSrc = data.result;

  // Chain hair color on top if enabled
  if (document.getElementById('tog-haircolor') &&
      document.getElementById('tog-haircolor').checked) {
    const hairData = await apiPost('/hair_color', {
      image:      resultSrc,
      hair_color: hexToRgb(document.getElementById('pick-haircolor').value),
      opacity:    parseInt(document.getElementById('sl-haircolor').value),
    });
    resultSrc = hairData.result;
  }

  document.getElementById('makeupResultPh').style.display = 'none';
  document.getElementById('makeupResultImg').src = resultSrc;
  document.getElementById('makeupResultImg').style.display = 'block';
  window.lastTransformedSrc = resultSrc;

  document.getElementById('sb-makeup').textContent = '✓';
  document.getElementById('sb-makeup').className = 'sidebar-badge done';
}