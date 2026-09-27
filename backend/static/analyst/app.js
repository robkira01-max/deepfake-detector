/* ── DeepfakeDetector Canada — Analyst UI JS ──────────────────────────────── */
'use strict';

// ── Gauge SVG (semicircle arc, r=85) ──────────────────────────────────────────
const GAUGE_ARC_LENGTH = Math.PI * 85; // ≈ 267.04

function initGauge(score) {
  const arcEl = document.getElementById('gauge-arc');
  if (!arcEl) return;

  arcEl.style.strokeDasharray = GAUGE_ARC_LENGTH;
  arcEl.style.strokeDashoffset = GAUGE_ARC_LENGTH; // hidden

  requestAnimationFrame(() => {
    setTimeout(() => {
      arcEl.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(0.23, 1, 0.32, 1)';
      arcEl.style.strokeDashoffset = GAUGE_ARC_LENGTH * (1 - score);
    }, 150);
  });

  // Counter — delayed to match arc start
  const scoreEl = document.getElementById('gauge-pct');
  if (scoreEl) {
    setTimeout(() => {
      let cur = 0;
      const target = Math.round(score * 100);
      const duration = 1300;
      const fps = 60;
      const step = target / (duration / (1000 / fps));
      const iv = setInterval(() => {
        cur = Math.min(cur + step, target);
        scoreEl.textContent = Math.round(cur) + '%';
        if (cur >= target) clearInterval(iv);
      }, 1000 / fps);
    }, 150);
  }
}

// ── Radar Chart (Chart.js) ────────────────────────────────────────────────────
function initRadarChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === 'undefined') return;

  const labels  = JSON.parse(canvas.getAttribute('data-labels') || '[]');
  const scores  = JSON.parse(canvas.getAttribute('data-scores') || '[]');
  const accentColor = canvas.getAttribute('data-color') || '#00cba4';

  new Chart(canvas, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        data: scores.map(s => Math.round(s * 100)),
        backgroundColor: accentColor + '22',
        borderColor: accentColor,
        borderWidth: 2,
        pointBackgroundColor: accentColor,
        pointBorderColor: '#0c1827',
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 1200, easing: 'easeInOutQuart' },
      scales: {
        r: {
          min: 0, max: 100,
          beginAtZero: true,
          ticks: {
            stepSize: 25,
            color: '#2d4460',
            font: { size: 9 },
            backdropColor: 'transparent',
          },
          grid: { color: '#1a2e47' },
          angleLines: { color: '#1a2e47' },
          pointLabels: {
            color: '#5a7a99',
            font: { size: 11, weight: '600' },
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0c1827',
          borderColor: '#1a2e47',
          borderWidth: 1,
          titleColor: '#dde8f5',
          bodyColor: '#5a7a99',
          callbacks: {
            label: ctx => ` ${ctx.raw}%`,
          },
        },
      },
    },
  });
}

// ── Score bar animations ──────────────────────────────────────────────────────
function initScoreBars() {
  const bars = document.querySelectorAll('.score-bar-fill');
  if (!bars.length) return;

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const bar = entry.target;
      const score = parseFloat(bar.getAttribute('data-score') || '0');
      bar.style.width = '0%';
      requestAnimationFrame(() => {
        setTimeout(() => {
          bar.style.transition = 'width 1s cubic-bezier(0.23, 1, 0.32, 1)';
          bar.style.width = Math.round(score * 100) + '%';
        }, 100);
      });
      observer.unobserve(bar);
    });
  }, { threshold: 0.1 });

  bars.forEach(b => observer.observe(b));
}

// ── Drop zone enhancement ─────────────────────────────────────────────────────
function initDropZone(zoneId) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;

  ['dragenter', 'dragover'].forEach(evt => {
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add('dragover'); });
  });
  ['dragleave', 'dragend'].forEach(evt => {
    zone.addEventListener(evt, () => zone.classList.remove('dragover'));
  });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const input = zone.querySelector('input[type=file]');
    if (input) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    }
    _updateFileName(zone, file.name, file.size);
  });

  const input = zone.querySelector('input[type=file]');
  if (input) {
    input.addEventListener('change', () => {
      const f = input.files?.[0];
      if (f) _updateFileName(zone, f.name, f.size);
    });
  }
}

function _updateFileName(zone, name, size) {
  let el = zone.querySelector('.drop-filename');
  if (!el) {
    el = document.createElement('span');
    el.className = 'drop-filename';
    zone.querySelector('.drop-subtitle')?.after(el);
  }
  const mb = (size / 1024 / 1024).toFixed(1);
  el.textContent = `${name}  (${mb} MB)`;
}

// ── HTMX polling stop on completion ──────────────────────────────────────────
document.addEventListener('htmx:afterSwap', evt => {
  const target = evt.detail.target;
  if (target && target.getAttribute('hx-trigger') === 'every 2s') {
    const statusEl = target.querySelector('[data-status]');
    if (!statusEl) return;
    const status = statusEl.getAttribute('data-status');
    if (status === 'completed' || status === 'failed') {
      // Stop polling, redirect to results
      target.removeAttribute('hx-trigger');
      const url = statusEl.getAttribute('data-results-url');
      if (url && status === 'completed') {
        setTimeout(() => { window.location.href = url; }, 600);
      }
    }
  }
});

// ── Auto-dismiss flash messages ───────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity 250ms cubic-bezier(0.23, 1, 0.32, 1), transform 250ms cubic-bezier(0.23, 1, 0.32, 1)';
    el.style.opacity = '0';
    el.style.transform = 'translateY(-6px)';
    setTimeout(() => el.remove(), 260);
  }, 5000);
});

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initScoreBars();
  initDropZone('upload-zone');

  // Results page
  const gaugeArc = document.getElementById('gauge-arc');
  if (gaugeArc) {
    const score = parseFloat(gaugeArc.getAttribute('data-score') || '0');
    initGauge(score);
  }
  initRadarChart('radar-chart');
});
