// js/expression.js
function toggleControl(name) {
  const tog = document.getElementById('tog-' + name);
  const sl  = document.getElementById('sl-' + name);
  const cc  = document.getElementById('cc-' + name);
  sl.disabled = !tog.checked;
  cc.classList.toggle('active-control', tog.checked);
  if (tog.checked && sl.value === '0') sl.value = 50;
  document.getElementById('sv-' + name).textContent = sl.value + '%';
}

function resetExpression() {
  ['smile','eyebrow','lips','slim'].forEach(n => {
    document.getElementById('tog-'+n).checked = false;
    document.getElementById('sl-'+n).value = 0;
    document.getElementById('sl-'+n).disabled = true;
    document.getElementById('sv-'+n).textContent = '0%';
    document.getElementById('cc-'+n).classList.remove('active-control');
  });
  document.getElementById('warpResultImg').style.display = 'none';
  document.getElementById('warpResultPh').style.display = 'flex';
}

async function runExpression() {
  if (!window.uploadedImageSrc) { alert('Upload an image first.'); return; }
  const body = {
    image:   window.uploadedImageSrc,
    smile:   document.getElementById('tog-smile').checked   ? document.getElementById('sl-smile').value   : 0,
    eyebrow: document.getElementById('tog-eyebrow').checked ? document.getElementById('sl-eyebrow').value : 0,
    lips:    document.getElementById('tog-lips').checked    ? document.getElementById('sl-lips').value    : 0,
    slim:    document.getElementById('tog-slim').checked    ? document.getElementById('sl-slim').value    : 0,
  };
  const data = await apiPost('/expression', body);
  document.getElementById('warpResultPh').style.display = 'none';
  document.getElementById('warpResultImg').src = data.result;
  document.getElementById('warpResultImg').style.display = 'block';
  window.lastTransformedSrc = data.result;
  document.getElementById('ps3').className = 'step-dot step-done';
  document.getElementById('ps3').textContent = '✓';
  document.getElementById('sb-expression').textContent = '✓';
  document.getElementById('sb-expression').className = 'sidebar-badge done';
}
