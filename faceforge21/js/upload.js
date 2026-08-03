// js/upload.js
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    window.uploadedImageSrc = e.target.result;
    document.getElementById('infName').textContent = file.name.length > 14 ? file.name.substring(0,13)+'…' : file.name;
    document.getElementById('infFormat').textContent = file.type.split('/')[1].toUpperCase();
    document.getElementById('infSize').textContent = (file.size/1024).toFixed(0)+' KB';
    const img = new Image();
    img.onload = () => document.getElementById('infDims').textContent = img.width+'×'+img.height;
    img.src = window.uploadedImageSrc;

    const zone = document.getElementById('uploadZone');
    zone.classList.add('has-image');
    zone.innerHTML = `<img src="${window.uploadedImageSrc}" style="max-height:160px;border-radius:8px;object-fit:cover"><div style="font-size:12px;color:var(--green);margin-top:8px">✓ Image loaded — click to replace</div><input type="file" id="imageInput" accept="image/*" style="display:none" onchange="handleImageUpload(event)">`;

    document.getElementById('previewRow').style.display = 'grid';
    document.getElementById('originalImg').src = window.uploadedImageSrc;
    document.getElementById('processedImg').src = window.uploadedImageSrc;

    // Propagate to all pages
    ['warpOrigImg','agingOrigImg','goldenOrigImg','makeupOrigImg'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.src = window.uploadedImageSrc; el.style.display = 'block'; }
    });
    ['warpOrigPh','agingOrigPh','goldenOrigPh','makeupOrigPh'].forEach(id => {
      const el = document.getElementById(id); if (el) el.style.display = 'none';
    });

    document.getElementById('ps1').className = 'step-dot step-done';
    document.getElementById('ps1').textContent = '✓';
    document.getElementById('sb-upload').textContent = '✓';
    document.getElementById('sb-upload').className = 'sidebar-badge done';
  };
  reader.readAsDataURL(file);
}