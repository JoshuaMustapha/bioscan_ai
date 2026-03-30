// Mobile camera handler for BioScan AI.
// Requests rear camera via getUserMedia (facingMode: environment).
// Falls back to a file input with capture="environment" if getUserMedia
// is unavailable (older Android WebViews, iOS Safari < 11, etc.).
// User previews the captured photo before submitting to POST /analyze.

(function () {
  'use strict';

  // Must match api/schemas.py LowConfidenceWarning exactly.
  var LOW_CONFIDENCE_WARNING =
    'This result has low confidence. Try retaking the photo with your full ' +
    'body visible and facing the camera directly.';

  // ── Element references ───────────────────────────────────────────────────
  var video           = document.getElementById('mobile-video');
  var canvas          = document.getElementById('mobile-canvas');
  var countdown       = document.getElementById('countdown');
  var photoPreview    = document.getElementById('photo-preview');
  var cameraUnavail   = document.getElementById('camera-unavailable');
  var fileLabel       = document.getElementById('file-label');
  var fileInput       = document.getElementById('file-input');
  var takePhotoBtn    = document.getElementById('take-photo-btn');
  var analyzeBtn      = document.getElementById('analyze-btn');
  var retakeBtn       = document.getElementById('retake-btn');
  var errorBox        = document.getElementById('error-box');
  var loading         = document.getElementById('loading');
  var results         = document.getElementById('results');

  var resultWeight  = document.getElementById('result-weight');
  var ciBarFill     = document.getElementById('ci-bar-fill');
  var ciBarMarker   = document.getElementById('ci-bar-marker');
  var ciBarMin      = document.getElementById('ci-bar-min');
  var ciBarMax      = document.getElementById('ci-bar-max');
  var ciRangeText   = document.getElementById('ci-range-text');
  var resultStd     = document.getElementById('result-std');
  var resultTime    = document.getElementById('result-time');
  var warningBox    = document.getElementById('low-confidence-warning');

  // Holds the captured image for submission: either a canvas Blob (stream
  // capture) or a File object (file input fallback).
  var _capturedBlob = null;

  // ── Camera init ──────────────────────────────────────────────────────────
  function initCamera() {
    var hasGetUserMedia =
      !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

    if (!hasGetUserMedia) {
      enableFileInputFallback();
      return;
    }

    navigator.mediaDevices
      .getUserMedia({
        // ideal: environment requests rear camera; falls back to any camera.
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      })
      .then(function (stream) {
        video.srcObject = stream;
        video.hidden = false;
        takePhotoBtn.hidden = false;
      })
      .catch(function () {
        // Permission denied or no camera — fall back to file input.
        enableFileInputFallback();
      });
  }

  function enableFileInputFallback() {
    video.hidden = true;
    cameraUnavail.hidden = false;
    fileLabel.hidden = false;
    takePhotoBtn.hidden = true;
  }

  // ── Countdown ────────────────────────────────────────────────────────────
  function runCountdown(callback) {
    var count = 3;
    takePhotoBtn.disabled = true;
    countdown.textContent = count;
    countdown.hidden      = false;

    var interval = setInterval(function () {
      count -= 1;
      if (count > 0) {
        countdown.textContent = count;
      } else {
        clearInterval(interval);
        countdown.hidden = true;
        callback();
      }
    }, 1000);
  }

  // ── Capture from live stream ─────────────────────────────────────────────
  function captureFromStream() {
    if (!video.videoWidth || !video.videoHeight) {
      showError('Camera stream is not ready yet. Wait a moment and try again.');
      return;
    }

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(
      function (blob) {
        if (!blob) {
          showError('Failed to encode camera frame. Please try again.');
          return;
        }
        _capturedBlob = blob;
        showPreview(URL.createObjectURL(blob));
      },
      'image/jpeg',
      0.92
    );
  }

  // ── File input fallback handler ──────────────────────────────────────────
  fileInput.addEventListener('change', function () {
    var file = fileInput.files && fileInput.files[0];
    if (!file) { return; }
    _capturedBlob = file;
    showPreview(URL.createObjectURL(file));
  });

  // ── Preview ──────────────────────────────────────────────────────────────
  function showPreview(objectUrl) {
    // Stop and hide the live video feed to save battery.
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(function (t) { t.stop(); });
    }
    video.hidden = true;
    cameraUnavail.hidden = true;
    fileLabel.hidden = true;
    takePhotoBtn.hidden = true;

    photoPreview.src = objectUrl;
    photoPreview.hidden = false;

    analyzeBtn.hidden = false;
    retakeBtn.hidden  = false;
    hideError();
    hideResults();
  }

  function resetToCamera() {
    // Revoke previous object URL to free memory.
    if (photoPreview.src && photoPreview.src.startsWith('blob:')) {
      URL.revokeObjectURL(photoPreview.src);
    }
    photoPreview.src = '';
    photoPreview.hidden = true;
    analyzeBtn.hidden = true;
    retakeBtn.hidden  = true;
    _capturedBlob = null;

    // Re-init camera (re-requests stream).
    initCamera();
  }

  // ── Submit ───────────────────────────────────────────────────────────────
  function submit() {
    if (!_capturedBlob) {
      showError('No photo captured. Please take or select a photo first.');
      return;
    }

    var heightCm = parseFloat(document.getElementById('height-cm').value);
    var age      = parseInt(document.getElementById('age').value, 10);
    var gender   = parseInt(document.getElementById('gender').value, 10);

    if (!heightCm || isNaN(heightCm) || heightCm < 50 || heightCm > 300) {
      showError('Please enter a valid height between 50 and 300 cm.');
      return;
    }
    if (!age || isNaN(age) || age < 5 || age > 120) {
      showError('Please enter a valid age between 5 and 120 years.');
      return;
    }

    var fd = new FormData();
    // _capturedBlob is either a Blob (canvas) or a File (input); both work.
    fd.append('image', _capturedBlob, 'photo.jpg');
    fd.append('height_cm', heightCm);
    fd.append('age', age);
    fd.append('gender', gender);

    setLoading(true);
    hideError();

    fetch('/analyze', { method: 'POST', body: fd })
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, status: resp.status, data: data };
        });
      })
      .then(function (r) {
        setLoading(false);
        if (!r.ok) {
          showError(r.data.detail || ('Server error (' + r.status + ')'));
          return;
        }
        renderResults(r.data);
      })
      .catch(function (err) {
        setLoading(false);
        showError('Network error — is the server reachable? (' + err.message + ')');
      });
  }

  // ── Render results ───────────────────────────────────────────────────────
  function renderResults(data) {
    resultWeight.textContent = data.estimated_weight_kg.toFixed(1) + ' kg';
    resultStd.textContent    = data.prediction_std.toFixed(2) + ' kg';
    resultTime.textContent   = data.processing_time_ms.toFixed(0) + ' ms';

    var low  = data.confidence_interval_low;
    var high = data.confidence_interval_high;
    var est  = data.estimated_weight_kg;

    ciRangeText.innerHTML =
      '<strong>' + low.toFixed(1) + ' – ' + high.toFixed(1) + ' kg</strong>';

    renderCIBar(est, low, high, data.prediction_std);

    if (data.low_confidence) {
      warningBox.textContent = LOW_CONFIDENCE_WARNING;
      warningBox.hidden = false;
    } else {
      warningBox.hidden = true;
    }

    results.hidden = false;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderCIBar(est, low, high, std) {
    var padding   = Math.max(std * 2, 5);
    var trackMin  = Math.max(0, low - padding);
    var trackMax  = high + padding;
    var range     = trackMax - trackMin;

    if (range <= 0) { range = 1; }

    var fillLeftPct   = ((low  - trackMin) / range * 100).toFixed(2);
    var fillWidthPct  = ((high - low)      / range * 100).toFixed(2);
    var markerLeftPct = ((est  - trackMin) / range * 100).toFixed(2);

    ciBarFill.style.left   = fillLeftPct + '%';
    ciBarFill.style.width  = fillWidthPct + '%';
    ciBarMarker.style.left = markerLeftPct + '%';

    ciBarMin.textContent = trackMin.toFixed(0) + ' kg';
    ciBarMax.textContent = trackMax.toFixed(0) + ' kg';
  }

  // ── UI helpers ───────────────────────────────────────────────────────────
  function setLoading(on) {
    loading.hidden      = !on;
    analyzeBtn.disabled = on;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
  }

  function hideError()   { errorBox.hidden = true; }
  function hideResults() { results.hidden  = true; }

  // ── Event listeners ──────────────────────────────────────────────────────
  takePhotoBtn.addEventListener('click', function () {
    runCountdown(captureFromStream);
  });
  analyzeBtn.addEventListener('click', submit);
  retakeBtn.addEventListener('click', resetToCamera);

  // ── Boot ─────────────────────────────────────────────────────────────────
  initCamera();

}());
