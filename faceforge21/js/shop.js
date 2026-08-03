// js/shop.js — Virtual Try-On Shop

// ── Product catalog ───────────────────────────────────────────────────────────
const PRODUCTS = {
  glasses: [
    { id: 'g1', name: 'Circle Sunnies',  price: '$240', emoji: '⭕', category: 'glasses',   asset: 'glasses/g1' },
    { id: 'g2', name: 'Gold Aviators',   price: '$320', emoji: '🕶',  category: 'glasses',   asset: 'glasses/g2' },
    { id: 'g3', name: 'Cat Eye Rose',    price: '$195', emoji: '😼', category: 'glasses',   asset: 'glasses/g3' },
    { id: 'g4', name: 'Thick Black',     price: '$180', emoji: '🔲', category: 'glasses',   asset: 'glasses/g4' },
  ],
  earrings: [
    { id: 'e1', name: 'Gold Studs',      price: '$85',  emoji: '🟡', category: 'earrings',  asset: 'earrings/e1' },
    { id: 'e2', name: 'Crystal Drops',   price: '$140', emoji: '💠', category: 'earrings',  asset: 'earrings/e2' },
    { id: 'e3', name: 'Gold Hoops',      price: '$110', emoji: '⭕', category: 'earrings',  asset: 'earrings/e3' },
    { id: 'e4', name: 'Pearl Dangles',   price: '$165', emoji: '🔵', category: 'earrings',  asset: 'earrings/e4' },
  ],
  necklaces: [
    { id: 'n1', name: 'Gold Chain',      price: '$280', emoji: '🔗', category: 'necklaces', asset: 'necklaces/n1' },
    { id: 'n2', name: 'Diamond Pendant', price: '$450', emoji: '💎', category: 'necklaces', asset: 'necklaces/n2' },
    { id: 'n3', name: 'Pearl Strand',    price: '$320', emoji: '📿', category: 'necklaces', asset: 'necklaces/n3' },
    { id: 'n4', name: 'Velvet Choker',   price: '$95',  emoji: '🎀', category: 'necklaces', asset: 'necklaces/n4' },
  ],
};

// ── State ─────────────────────────────────────────────────────────────────────
const ASSETS        = {};      // { 'glasses/g1': HTMLImageElement, ... }
let shopStream      = null;
let faceLandmarker  = null;
let mpLoading       = false;
let detectionRunning = false;
let selectedProduct = null;
let cart            = [];
let currentCategory = 'glasses';
let shopInitialized = false;

// null = not yet detected; true = camera delivers already-mirrored (selfie) frames;
// false = camera delivers raw (non-mirrored) frames that we must flip ourselves.
let cameraIsMirrored = null;

let hairSegmenter  = null;
let hairConfData   = null;   // Float32Array confidence at model resolution
let hairConfW      = 0;
let hairConfH      = 0;
let hairBBox       = null;   // {minX,minY,maxX,maxY,w,cx} in canvas px
let maskCanvas     = null;   // hair confidence drawn as canvas for compositing
let offCanvas      = null;   // face-layer offscreen canvas
let offCtx         = null;
let segFrameCount  = 7;      // start at 7 so first segmentation fires on frame 1

// ── White-background removal ──────────────────────────────────────────────────
// Zeroes alpha for pixels that are bright AND near-grey (actual white background).
// Preserves gold, pearl, coloured pixels even if they are light.
function stripWhiteBg(img) {
  return new Promise(resolve => {
    const oc  = document.createElement('canvas');
    oc.width  = img.naturalWidth  || 256;
    oc.height = img.naturalHeight || 256;
    const ctx = oc.getContext('2d');
    ctx.drawImage(img, 0, 0);
    let id;
    try { id = ctx.getImageData(0, 0, oc.width, oc.height); }
    catch (_) { resolve(img); return; }   // CORS taint — use image as-is

    const d = id.data;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const lo  = Math.min(r, g, b);       // brightness of darkest channel
      const sat = Math.max(r, g, b) - lo;  // colour saturation
      if (lo > 230 && sat < 30) {
        // Fully transparent: white/near-grey background
        d[i + 3] = 0;
      } else if (lo > 200 && sat < 20) {
        // Soft semi-transparent edge for anti-aliased fringe pixels
        d[i + 3] = Math.round(((lo - 200) / 30) * (255 - d[i + 3]));
      }
    }
    ctx.putImageData(id, 0, 0);

    const out = new Image();
    out.onload  = () => resolve(out);
    out.onerror = () => resolve(img);
    out.src = oc.toDataURL('image/png');
  });
}

// ── Asset loading (served through backend to avoid file:// CORS issues) ───────
async function loadAssets() {
  const BASE = 'http://localhost:5050/assets';
  const all  = Object.values(PRODUCTS).flat();

  await Promise.all(all.map(p => new Promise(res => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = async () => {
      ASSETS[p.asset] = await stripWhiteBg(img);
      res();
    };
    img.onerror = () => res();
    img.src = `${BASE}/${p.asset}.png`;
  })));
}

// ── MediaPipe face landmarker (dynamic import — camera doesn't wait) ──────────
async function initMediaPipe() {
  if (faceLandmarker || mpLoading) return;
  mpLoading = true;
  try {
    const { FaceLandmarker, ImageSegmenter, FilesetResolver } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );
    const resolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: 'http://localhost:5050/api/model/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
    console.log('MediaPipe face landmarker ready ✓');
    // Load hair segmenter in parallel — failure is non-fatal (wigs fall back to landmarks)
    try {
      hairSegmenter = await ImageSegmenter.createFromOptions(resolver, {
        baseOptions: {
          modelAssetPath: 'http://localhost:5050/api/model/hair_segmenter.tflite',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
      console.log('Hair segmenter ready ✓');
    } catch (e) {
      console.warn('Hair segmenter optional — wigs use landmark fallback:', e.message);
    }
  } catch (e) {
    console.error('MediaPipe init failed:', e);
  }
  mpLoading = false;
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startShopCamera() {
  const video  = document.getElementById('shopVideo');
  const canvas = document.getElementById('shopCanvas');
  if (shopStream || !video || !canvas) return;

  try {
    shopStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = shopStream;

    video.addEventListener('loadeddata', () => {
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;

      // Hide placeholder once camera is live
      const ph = document.getElementById('shopVideoPlaceholder');
      if (ph) ph.style.display = 'none';

      if (!detectionRunning) { detectionRunning = true; renderLoop(); }
    }, { once: true });

  } catch (e) {
    showShopStatus('Camera access denied — ' + e.message, '#f75f5f');
  }
}

function stopShopCamera() {
  detectionRunning = false;
  cameraIsMirrored = null;  // re-detect on next camera start
  if (shopStream) { shopStream.getTracks().forEach(t => t.stop()); shopStream = null; }
  const video = document.getElementById('shopVideo');
  if (video) video.srcObject = null;
  const ph = document.getElementById('shopVideoPlaceholder');
  if (ph) ph.style.display = 'flex';
}

// ── Camera orientation detection ──────────────────────────────────────────────
// Some cameras (iOS Safari, some Android) deliver already-mirrored (selfie) frames
// to canvas APIs; others (Chrome desktop) deliver raw unmirrored frames.
// We detect which by checking which side MediaPipe puts the person's LEFT eye on:
//   - Raw frame:      lm33.x > 0.5  (left eye appears on RIGHT half of raw image)
//   - Mirrored frame: lm33.x < 0.5  (left eye appears on LEFT half of selfie image)
function detectCameraOrientation(landmarks) {
  if (cameraIsMirrored !== null) return;
  const p33  = landmarks[33];
  const p263 = landmarks[263];
  if (!p33 || !p263) return;
  // lm33 = person's LEFT eye, lm263 = person's RIGHT eye.
  // In a raw frame the left eye is at higher x (right side), so lm33.x > lm263.x.
  // In a mirrored frame the left eye is at lower x (left side), so lm33.x < lm263.x.
  cameraIsMirrored = p33.x < p263.x;
}

// ── Hair segmentation helpers ─────────────────────────────────────────────────
function computeHairBBox(cw, ch) {
  if (!hairConfData || !hairConfW || !hairConfH) return;
  let minX = hairConfW, minY = hairConfH, maxX = 0, maxY = 0, count = 0;
  for (let i = 0; i < hairConfData.length; i++) {
    if (hairConfData[i] > 0.4) {
      const x = i % hairConfW;
      const y = (i / hairConfW) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      count++;
    }
  }
  if (count > 300) {
    hairBBox = {
      minX: minX / hairConfW * cw,
      minY: minY / hairConfH * ch,
      maxX: maxX / hairConfW * cw,
      maxY: maxY / hairConfH * ch,
      w:    (maxX - minX) / hairConfW * cw,
      cx:   ((minX + maxX) / 2) / hairConfW * cw,
    };
  }
}

// Renders hair confidence floats into a small canvas used as a destination-out mask.
function updateHairMaskCanvas() {
  if (!hairConfData || !hairConfW || !hairConfH) return;
  if (!maskCanvas) maskCanvas = document.createElement('canvas');
  maskCanvas.width  = hairConfW;
  maskCanvas.height = hairConfH;
  const mCtx = maskCanvas.getContext('2d');
  const id   = new ImageData(hairConfW, hairConfH);
  const d    = id.data;
  for (let i = 0; i < hairConfData.length; i++) {
    const conf = hairConfData[i];
    // Smooth threshold: fully transparent <0.30, fully opaque >0.70
    d[i * 4 + 3] = conf < 0.30 ? 0 : conf > 0.70 ? 255 : ((conf - 0.30) / 0.40 * 255) | 0;
  }
  mCtx.putImageData(id, 0, 0);
}

// Draws the face layer (video with hair pixels cut out) on top of the wig,
// so the face edges appear in front of the wig — the Snapchat compositing trick.
function applyFaceOverWig(ctx, video, cw, ch) {
  if (!maskCanvas) return;
  if (!offCanvas) {
    offCanvas = document.createElement('canvas');
    offCtx    = offCanvas.getContext('2d');
  }
  if (offCanvas.width !== cw || offCanvas.height !== ch) {
    offCanvas.width = cw; offCanvas.height = ch;
  } else {
    offCtx.clearRect(0, 0, cw, ch);
  }

  // Draw mirrored video to face layer
  if (cameraIsMirrored) {
    offCtx.drawImage(video, 0, 0, cw, ch);
  } else {
    offCtx.save();
    offCtx.scale(-1, 1);
    offCtx.drawImage(video, -cw, 0, cw, ch);
    offCtx.restore();
  }

  // Cut hair region from face layer — wig shows through where hair was
  offCtx.save();
  offCtx.globalCompositeOperation = 'destination-out';
  if (!cameraIsMirrored) {
    // Mask is in raw camera space; face layer was drawn mirrored, so flip mask too
    offCtx.translate(cw, 0);
    offCtx.scale(-1, 1);
  }
  offCtx.drawImage(maskCanvas, 0, 0, cw, ch);
  offCtx.restore();

  // Composite: face (non-hair) pixels sit on top of the wig
  ctx.drawImage(offCanvas, 0, 0);
}

// ── Render loop ───────────────────────────────────────────────────────────────
function renderLoop() {
  if (!detectionRunning) return;
  const video  = document.getElementById('shopVideo');
  const canvas = document.getElementById('shopCanvas');
  if (!video || !canvas) return;
  const ctx = canvas.getContext('2d');

  if (video.readyState >= 2) {
    // Run landmark detection every frame until we know the camera orientation,
    // then only run it when a product is being shown (saves CPU).
    let faceResult = null;
    const needDetection = faceLandmarker && (cameraIsMirrored === null || selectedProduct);
    if (needDetection) {
      try {
        const res = faceLandmarker.detectForVideo(video, performance.now());
        if (res.faceLandmarks && res.faceLandmarks.length > 0) {
          faceResult = res.faceLandmarks[0];
          detectCameraOrientation(faceResult);
        }
      } catch (_) {}
    }

    // Run hair segmentation every 8 frames when a wig is active
    if (hairSegmenter && selectedProduct?.category === 'wigs' && ++segFrameCount % 8 === 0) {
      try {
        const seg = hairSegmenter.segmentForVideo(video, performance.now());
        const m   = seg.confidenceMasks?.[0];
        if (m) {
          hairConfData = m.getAsFloat32Array().slice();
          hairConfW    = m.width;
          hairConfH    = m.height;
          m.close();
          computeHairBBox(canvas.width, canvas.height);
          updateHairMaskCanvas();
        }
      } catch(_) {}
    }

    // Draw video: if camera already delivers a selfie/mirrored frame, draw directly;
    // otherwise flip it with ctx.scale so the canvas shows a proper selfie view.
    if (cameraIsMirrored) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();
    }

    // Draw accessory overlay; wigs get face-over-wig compositing for realism
    if (faceResult && selectedProduct) {
      drawProduct(ctx, faceResult, canvas.width, canvas.height);
      if (selectedProduct.category === 'wigs' && maskCanvas) {
        applyFaceOverWig(ctx, video, canvas.width, canvas.height);
      }
    }
  }
  requestAnimationFrame(renderLoop);
}

// ── Landmark helper ───────────────────────────────────────────────────────────
// Always returns coordinates in selfie display space (person's left = canvas left).
// If the camera delivers raw frames, we flip x.  If already mirrored, we use x as-is.
function lm(landmarks, idx, cw, ch) {
  const p = landmarks[idx];
  if (!p) return { x: 0, y: 0 };
  const x = cameraIsMirrored ? p.x * cw : (1 - p.x) * cw;
  return { x, y: p.y * ch };
}

// ── Head pose ─────────────────────────────────────────────────────────────────
// Uses lm() mirrored positions.  In selfie space: rCx > lCx (right eye is RIGHT).
// roll = atan2(rCy-lCy, rCx-lCx) → 0 when face is level; positive = tilt right (CW).
function getHeadPose(landmarks, cw, ch) {
  const lOuter = lm(landmarks, 33,  cw, ch);
  const lInner = lm(landmarks, 133, cw, ch);
  const rInner = lm(landmarks, 362, cw, ch);
  const rOuter = lm(landmarks, 263, cw, ch);

  const lCx = (lOuter.x + lInner.x) / 2;
  const lCy = (lOuter.y + lInner.y) / 2;
  const rCx = (rInner.x + rOuter.x) / 2;
  const rCy = (rInner.y + rOuter.y) / 2;

  // rCx > lCx in selfie coords → denominator positive → atan2(0,pos) = 0 ✓
  const roll = Math.atan2(rCy - lCy, rCx - lCx);

  return { roll, lCx, lCy, rCx, rCy };
}

// ── Earlobe estimation ────────────────────────────────────────────────────────
function earlobes(landmarks, cw, ch) {
  const lJaw = lm(landmarks, 234, cw, ch);
  const rJaw = lm(landmarks, 454, cw, ch);
  const fw   = Math.abs(rJaw.x - lJaw.x);
  // lm234/454 sit at the tragus (ear-face junction); the lobe hangs ~10% of face-width lower
  const lobeOffset = fw * 0.10;
  return {
    left:  { x: lJaw.x, y: lJaw.y + lobeOffset },
    right: { x: rJaw.x, y: rJaw.y + lobeOffset },
    fw,
  };
}

// ── Neck geometry ─────────────────────────────────────────────────────────────
function neckGeom(landmarks, cw, ch) {
  const chin = lm(landmarks, 152, cw, ch);
  const lJaw = lm(landmarks, 234, cw, ch);
  const rJaw = lm(landmarks, 454, cw, ch);
  const fw   = Math.abs(rJaw.x - lJaw.x);
  return { chin, fw, neckY: chin.y + fw * 0.22 };
}

// ── Core PNG drawing helper ───────────────────────────────────────────────────
// ctx.scale(-1,1) inside mirrors the PNG to match the selfie-flipped video.
function placePNG(ctx, img, cx, cy, drawW, roll, alpha = 0.95) {
  if (!img || !img.complete || !img.naturalWidth) return;
  const drawH = drawW * (img.naturalHeight / img.naturalWidth);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(cx, cy);
  ctx.rotate(roll);
  ctx.scale(-1, 1);                                          // mirror to match selfie video
  ctx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
  ctx.restore();
}

// ── PNG drawing per category ──────────────────────────────────────────────────
function drawGlassesPNG(ctx, img, landmarks, cw, ch) {
  const { lCx, lCy, rCx, rCy, roll } = getHeadPose(landmarks, cw, ch);
  const lOuter = lm(landmarks, 33,  cw, ch);
  const rOuter = lm(landmarks, 263, cw, ch);

  const cx    = (lCx + rCx) / 2;
  const cy    = (lCy + rCy) / 2;
  // 1.9× eye-outer span spans the full frame including temples
  const width = Math.abs(rOuter.x - lOuter.x) * 1.9;
  placePNG(ctx, img, cx, cy, width, roll);
}

function drawEarringsPNG(ctx, img, landmarks, cw, ch) {
  if (!img || !img.complete || !img.naturalWidth) return;
  const { left, right, fw } = earlobes(landmarks, cw, ch);
  const { roll } = getHeadPose(landmarks, cw, ch);
  const earW = fw * 0.17;
  const earH = earW * (img.naturalHeight / img.naturalWidth);

  // Hide only when that ear is significantly rotated away (z > 0.3).
  // Front-facing z for lm234/454 is naturally 0.05-0.15, so 0.05 was too strict.
  const leftVisible  = (landmarks[234] ? landmarks[234].z : 0) < 0.3;
  const rightVisible = (landmarks[454] ? landmarks[454].z : 0) < 0.3;

  // Earrings hang DOWN from the lobe (y=0 = top of image = lobe attachment).
  // Right earring is mirrored for selfie symmetry.
  if (leftVisible) {
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.translate(left.x, left.y);
    ctx.rotate(roll);
    ctx.drawImage(img, -earW / 2, 0, earW, earH);
    ctx.restore();
  }
  if (rightVisible) {
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.translate(right.x, right.y);
    ctx.rotate(roll);
    ctx.scale(-1, 1);
    ctx.drawImage(img, -earW / 2, 0, earW, earH);
    ctx.restore();
  }
}

function drawNecklacePNG(ctx, img, landmarks, cw, ch) {
  if (!img || !img.complete || !img.naturalWidth) return;
  const { chin, fw, neckY } = neckGeom(landmarks, cw, ch);
  const { roll } = getHeadPose(landmarks, cw, ch);
  const drawW = fw * 1.15;
  const drawH = drawW * (img.naturalHeight / img.naturalWidth);
  // Top-anchor: necklace top starts at neckY (just below chin), hangs down.
  // placePNG centers at cy, so shift cy down by half the image height.
  placePNG(ctx, img, chin.x, neckY + drawH / 2, drawW, roll);
}

function drawWigPNG(ctx, img, landmarks, cw, ch) {
  if (!img || !img.complete || !img.naturalWidth) return;
  const forehead = lm(landmarks, 10,  cw, ch);
  const lFace    = lm(landmarks, 234, cw, ch);
  const rFace    = lm(landmarks, 454, cw, ch);
  const chin     = lm(landmarks, 152, cw, ch);
  const { roll } = getHeadPose(landmarks, cw, ch);

  const fw    = Math.abs(rFace.x - lFace.x);
  const faceH = Math.abs(chin.y - forehead.y);

  let wigW, cx, topY;
  if (hairBBox && hairBBox.w > fw * 0.3) {
    // Segmentation-guided: size from detected hair region
    wigW = hairBBox.w * 1.55;
    cx   = hairBBox.cx;
    topY = hairBBox.minY - faceH * 0.18;
  } else {
    // Landmark fallback: wider wig placed well above the forehead
    wigW = fw * 2.5;
    cx   = (lFace.x + rFace.x) / 2;
    topY = forehead.y - faceH * 0.48;
  }

  const AR      = img.naturalHeight / img.naturalWidth;
  const srcFrac = AR > 1.5 ? 0.75 : 1.0;   // crop very tall images at 75%
  const srcH    = Math.floor(img.naturalHeight * srcFrac);
  const dispH   = wigW * (srcH / img.naturalWidth);
  const cy      = topY + dispH / 2;

  ctx.save();
  ctx.globalAlpha = 0.93;
  ctx.translate(cx, cy);
  ctx.rotate(roll);
  ctx.scale(-1, 1);
  ctx.drawImage(img, 0, 0, img.naturalWidth, srcH, -wigW / 2, -dispH / 2, wigW, dispH);
  ctx.restore();
}

// ── Product drawing dispatcher ────────────────────────────────────────────────
function drawProduct(ctx, landmarks, cw, ch) {
  if (!selectedProduct) return;
  const { category, asset } = selectedProduct;
  const img = ASSETS[asset];
  if (!img) return;

  if      (category === 'glasses')   drawGlassesPNG(ctx, img, landmarks, cw, ch);
  else if (category === 'earrings')  drawEarringsPNG(ctx, img, landmarks, cw, ch);
  else if (category === 'necklaces') drawNecklacePNG(ctx, img, landmarks, cw, ch);
  else if (category === 'wigs')      drawWigPNG(ctx, img, landmarks, cw, ch);
}

// ── Capture ───────────────────────────────────────────────────────────────────
function captureShopFrame() {
  const canvas = document.getElementById('shopCanvas');
  return canvas ? canvas.toDataURL('image/jpeg', 0.92) : null;
}

function captureShopPhoto() {
  const dataUrl = captureShopFrame();
  if (!dataUrl) { alert('Camera not active.'); return; }
  const area = document.getElementById('shopCaptureArea');
  if (!area) return;
  const existing = area.querySelectorAll('.shop-captured-img');
  if (existing.length >= 4) existing[0].remove();
  const img = document.createElement('img');
  img.src = dataUrl;
  img.className = 'shop-captured-img';
  img.title = `Captured at ${new Date().toLocaleTimeString()}`;
  area.appendChild(img);
  area.style.display = 'flex';
  window._lastShopCapture = dataUrl;
}

// ── Category switching ────────────────────────────────────────────────────────
function switchCategory(cat) {
  currentCategory = cat;
  document.querySelectorAll('.shop-cat-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll(`.shop-cat-btn[data-cat="${cat}"]`).forEach(b => b.classList.add('active'));
  const title = document.getElementById('shopGridTitle');
  const labels = { glasses: '👓 Glasses', earrings: '💎 Earrings', necklaces: '📿 Necklaces', wigs: '💇 Wigs' };
  if (title) title.textContent = labels[cat] || cat;
  renderProductGrid();
}

function renderProductGrid() {
  const grid = document.getElementById('shopProductGrid');
  if (!grid) return;
  const items = PRODUCTS[currentCategory] || [];
  grid.innerHTML = items.map(p => `
    <div class="shop-product-card ${selectedProduct && selectedProduct.id === p.id ? 'selected' : ''}"
         id="pcard-${p.id}" onclick="window.shopSelectProduct('${p.id}')">
      <div class="shop-product-emoji">${p.emoji}</div>
      <div class="shop-product-name">${p.name}</div>
      <div class="shop-product-price">${p.price}</div>
      <div class="shop-product-btns">
        <button class="shop-btn-cart" onclick="event.stopPropagation();window.shopAddToCart('${p.id}')">+ Cart</button>
      </div>
    </div>`).join('');
}

// ── Product selection ─────────────────────────────────────────────────────────
function selectProduct(id) {
  const all   = Object.values(PRODUCTS).flat();
  const found = all.find(p => p.id === id);
  if (!found) return;
  if (selectedProduct && selectedProduct.id === id) {
    selectedProduct = null;
  } else {
    selectedProduct = found;
  }
  document.querySelectorAll('.shop-product-card').forEach(c => c.classList.remove('selected'));
  if (selectedProduct) {
    const card = document.getElementById('pcard-' + selectedProduct.id);
    if (card) card.classList.add('selected');
  }
}

// ── Cart ──────────────────────────────────────────────────────────────────────
function addToCart(id) {
  const all  = Object.values(PRODUCTS).flat();
  const prod = all.find(p => p.id === id);
  if (!prod) return;
  if (cart.find(c => c.id === id)) { showShopStatus(`${prod.name} already in cart`, '#f7a24f'); return; }
  cart.push(prod);
  updateCartBadge();
  showShopStatus(`${prod.name} added ✓`, '#1fe89f');
  setTimeout(() => showShopStatus('', ''), 1800);
}

function removeFromCart(id) {
  cart = cart.filter(p => p.id !== id);
  updateCartBadge();
  renderCartPanel();
}

function updateCartBadge() {
  const badge = document.getElementById('shopCartBadge');
  if (badge) { badge.textContent = cart.length; badge.style.display = cart.length ? 'flex' : 'none'; }
}

function openCartPanel() {
  renderCartPanel();
  const panel = document.getElementById('shopCartPanel');
  if (panel) panel.classList.toggle('open');
}

function renderCartPanel() {
  const body = document.getElementById('shopCartBody');
  if (!body) return;
  if (cart.length === 0) {
    body.innerHTML = '<div style="color:var(--muted2);font-size:12px;text-align:center;padding:20px">Your cart is empty</div>';
    return;
  }
  const total = cart.reduce((s, p) => s + parseInt(p.price.replace('$', '')), 0);
  body.innerHTML = cart.map(p => `
    <div class="shop-cart-item">
      <span>${p.emoji} ${p.name}</span>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="color:var(--accent);font-family:'DM Mono',monospace;font-size:11px">${p.price}</span>
        <button onclick="window.shopRemoveFromCart('${p.id}')"
                style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px">✕</button>
      </div>
    </div>`).join('') + `
    <div style="border-top:1px solid var(--border);padding-top:12px;margin-top:8px;
                display:flex;justify-content:space-between;align-items:center">
      <span style="font-weight:600;color:var(--text)">Total</span>
      <span style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:var(--accent)">$${total}</span>
    </div>
    <button class="btn btn-primary" style="width:100%;margin-top:12px;justify-content:center"
            onclick="window.shopFaceCheckout()">👤 Face Checkout</button>`;
}

// ── Fal: Convince Me (uses flux-pro/kontext for identity-preserving gen) ──────
async function convinceMe(id) {
  const all  = Object.values(PRODUCTS).flat();
  const prod = all.find(p => p.id === id);
  if (!prod) return;
  const snap = captureShopFrame();
  showFalModal(`✨ Convincing you about ${prod.name}…`, null, true);
  try {
    const resp = await fetch('http://localhost:5050/api/shop/convince', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: snap, product: prod.name }),
    });
    const data = await resp.json();
    showFalModal(`✨ Why you NEED ${prod.name}`, data.image_url, false);
  } catch (e) {
    closeFalModal();
    alert('Fal generation failed — is backend running?');
  }
}

// ── Fal: Face Checkout ────────────────────────────────────────────────────────
async function faceCheckout() {
  if (cart.length === 0) { alert('Your cart is empty!'); return; }
  const snap      = captureShopFrame();
  const cartNames = cart.map(p => p.name);
  showFalModal('Creating your haul look…', null, true);
  try {
    const resp = await fetch('http://localhost:5050/api/shop/checkout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: snap, cart: cartNames }),
    });
    const data = await resp.json();
    showFalModal(`🛍 Your Haul: ${cartNames.join(', ')}`, data.image_url, false);
  } catch (e) {
    closeFalModal();
    alert('Checkout failed — is backend running?');
  }
}

// ── Fal modal ─────────────────────────────────────────────────────────────────
function showFalModal(title, imageUrl, loading) {
  const modal = document.getElementById('shopFalModal');
  if (!modal) return;
  modal.style.display = 'flex';
  document.getElementById('shopFalTitle').textContent = title;
  const img  = document.getElementById('shopFalImage');
  const spin = document.getElementById('shopFalSpinner');
  if (loading) { img.style.display = 'none'; spin.style.display = 'flex'; }
  else {
    spin.style.display = 'none';
    if (imageUrl) { img.src = imageUrl; img.style.display = 'block'; }
  }
}
function closeFalModal() {
  const modal = document.getElementById('shopFalModal');
  if (modal) modal.style.display = 'none';
}

// ── Status bar ────────────────────────────────────────────────────────────────
function showShopStatus(msg, color) {
  const el = document.getElementById('shopStatusBar');
  if (!el) return;
  el.textContent   = msg;
  el.style.color   = color || 'var(--muted2)';
  el.style.display = msg ? 'block' : 'none';
}

// ── Tab lifecycle ─────────────────────────────────────────────────────────────
async function onShopTabActivated() {
  if (!shopInitialized) {
    shopInitialized = true;
    switchCategory('glasses');
    loadAssets();          // load PNGs in background
    initMediaPipe();       // load face detection in background
  }
  await startShopCamera(); // camera starts immediately
}

function onShopTabDeactivated() {
  stopShopCamera();
}

// ── Window exports ────────────────────────────────────────────────────────────
window.shopSelectProduct    = selectProduct;
window.shopAddToCart        = addToCart;
window.shopRemoveFromCart   = removeFromCart;
window.shopConvinceMe       = convinceMe;
window.shopOpenCart         = openCartPanel;
window.shopFaceCheckout     = faceCheckout;
window.shopCapturePhoto     = captureShopPhoto;
window.shopSwitchCategory   = switchCategory;
window.shopCloseFalModal    = closeFalModal;
window.onShopTabActivated   = onShopTabActivated;
window.onShopTabDeactivated = onShopTabDeactivated;
