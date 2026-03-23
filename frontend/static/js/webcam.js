// Desktop webcam handler for BioScan AI.
// Initialises the MediaDevices stream on page load, captures a still frame
// on button press, and POSTs it with height/age to POST /analyze.

(function () {
  'use strict';

  // Must match api/schemas.py LowConfidenceWarning exactly.
  var LOW_CONFIDENCE_WARNING =
    'This result has low confidence. Try retaking the photo with your full ' +
    'body visible and facing the camera directly.';

  // ── Element references ───────────────────────────────────────────────────
  var video      = document.getElementById('webcam-video');
  var canvas     = document.getElementById('capture-canvas');
  var captureBtn = document.getElementById('capture-btn');
  var qrBtn      = document.getElementById('qr-btn');
  var qrSection  = document.getElementById('qr-section');
  var errorBox   = document.getElementById('error-box');
  var loading    = document.getElementById('loading');
  var results    = document.getElementById('results');

  var resultWeight   = document.getElementById('result-weight');
  var ciBarFill      = document.getElementById('ci-bar-fill');
  var ciBarMarker    = document.getElementById('ci-bar-marker');
  var ciBarMin       = document.getElementById('ci-bar-min');
  var ciBarMax       = document.getElementById('ci-bar-max');
  var ciRangeText    = document.getElementById('ci-range-text');
  var resultStd      = document.getElementById('result-std');
  var resultTime     = document.getElementById('result-time');
  var warningBox     = document.getElementById('low-confidence-warning');

  // ── Webcam init ──────────────────────────────────────────────────────────
  function initWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraDenied('getUserMedia is not supported in this browser.');
      return;
    }

    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      .then(function (stream) {
        video.srcObject = stream;
      })
      .catch(function (err) {
        showCameraDenied(
          'Camera access denied or unavailable. ' +
          'Allow camera permissions in your browser settings and reload the page. ' +
          '(' + err.name + ')'
        );
        captureBtn.disabled = true;
      });
  }

  function showCameraDenied(msg) {
    var el = document.getElementById('camera-denied');
    if (el) { el.textContent = msg; el.hidden = false; }
    video.hidden = true;
  }

  // ── Frame capture ────────────────────────────────────────────────────────
  function captureFrame() {
    return new Promise(function (resolve, reject) {
      if (!video.videoWidth || !video.videoHeight) {
        reject(new Error('Webcam stream has no dimensions yet. Try again in a moment.'));
        return;
      }
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      // Mirror the canvas to match the CSS-mirrored video preview.
      var ctx = canvas.getContext('2d');
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0);
      ctx.restore();

      canvas.toBlob(
        function (blob) {
          if (blob) { resolve(blob); }
          else { reject(new Error('Failed to encode canvas to JPEG.')); }
        },
        'image/jpeg',
        0.92
      );
    });
  }

  // ── Submit ───────────────────────────────────────────────────────────────
  function submit() {
    var heightCm = parseFloat(document.getElementById('height-cm').value);
    var age      = parseInt(document.getElementById('age').value, 10);

    if (!heightCm || isNaN(heightCm) || heightCm < 50 || heightCm > 300) {
      showError('Please enter a valid height between 50 and 300 cm.');
      return;
    }
    if (!age || isNaN(age) || age < 5 || age > 120) {
      showError('Please enter a valid age between 5 and 120 years.');
      return;
    }

    captureFrame()
      .then(function (blob) {
        var fd = new FormData();
        fd.append('image', blob, 'capture.jpg');
        fd.append('height_cm', heightCm);
        fd.append('age', age);

        setLoading(true);
        hideError();
        hideResults();

        return fetch('/analyze', { method: 'POST', body: fd });
      })
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
        if (err.message && err.message.indexOf('Webcam') === 0) {
          showError(err.message);
        } else {
          showError('Network error — is the server running? (' + err.message + ')');
        }
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
    var padding  = Math.max(std * 2, 5);
    var trackMin = Math.max(0, low - padding);
    var trackMax = high + padding;
    var range    = trackMax - trackMin;

    if (range <= 0) { range = 1; } // guard against degenerate case

    var fillLeftPct  = ((low  - trackMin) / range * 100).toFixed(2);
    var fillWidthPct = ((high - low)      / range * 100).toFixed(2);
    var markerLeftPct = ((est - trackMin) / range * 100).toFixed(2);

    ciBarFill.style.left   = fillLeftPct + '%';
    ciBarFill.style.width  = fillWidthPct + '%';
    ciBarMarker.style.left = markerLeftPct + '%';

    ciBarMin.textContent = trackMin.toFixed(0) + ' kg';
    ciBarMax.textContent = trackMax.toFixed(0) + ' kg';
  }

  // ── UI helpers ───────────────────────────────────────────────────────────
  function setLoading(on) {
    loading.hidden    = !on;
    captureBtn.disabled = on;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
  }

  function hideError()   { errorBox.hidden = true; }
  function hideResults() { results.hidden  = true; }

  // ── Event listeners ──────────────────────────────────────────────────────
  captureBtn.addEventListener('click', submit);

  qrBtn.addEventListener('click', function () {
    qrSection.hidden = !qrSection.hidden;
    if (!qrSection.hidden) {
      qrSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });

  // ── Boot ─────────────────────────────────────────────────────────────────
  initWebcam();

}());
