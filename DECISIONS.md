# Architectural Decisions — BioScan AI

## 1. Height as user input, not estimated
Estimating height from a 2D image without a calibration reference is unreliable due to perspective distortion. Height is collected from the user directly and passed through as input_height_cm.

## 2. MLP over CNN for weight estimation
Feature engineering in Stage 2 removes spatial noise from the image by converting it to geometric ratios. A CNN's spatial pattern detection adds no value at that point — pure geometric features feed better into an MLP.

## 3. Monte Carlo Dropout for confidence intervals
Chose MC Dropout over deep ensembles for simplicity of implementation while still producing calibrated uncertainty estimates.

## 4. ANSUR II as primary training dataset
Real anthropometric measurement-to-weight relationships. No images required — MediaPipe acts as the measurement extractor bridging image to ANSUR II's measurement space.

## 5. SMPL synthetic renders for pipeline validation
Used synthetic data generation to validate the full pipeline end-to-end before acquiring real-world data.

## 6. ngrok for mobile HTTPS in development
Mobile browsers block camera access on non-HTTPS origins. ngrok provides a public HTTPS tunnel in development without needing a deployed server.

## 7. static_image_mode=True in MediaPipe
Each API request is an independent still image, not a video stream. Setting static_image_mode=True prevents MediaPipe from carrying tracking state between unrelated requests, which would corrupt results in a multi-user API context.

## 8. Image size capped at 1280px longest side
Uncapped uploads cause memory exhaustion under concurrent load. 1280px is sufficient for MediaPipe accuracy and matches typical mobile JPEG output.

## 9. threading.Lock for MediaPipe thread safety
MediaPipe's TFLite interpreter is not documented as thread-safe. A Lock protects against silent result corruption under FastAPI's concurrent workers. Production scale would use a PoseDetector instance pool instead.

## 10. Ratios over raw measurements in feature engineering
MediaPipe landmarks are normalized to [0,1] relative to image dimensions, not real-world units. Raw distances are therefore perspective-dependent. Ratios (e.g. shoulder_to_hip_ratio) cancel out perspective distortion automatically and are stable features regardless of subject distance from camera.

## 11. Z coordinate excluded from Euclidean distance
MediaPipe's z value is a relative depth estimate, not metric depth. Mixing it into normalized x/y plane distances adds noise rather than information. All distance calculations use x and y only.

## 12. Visibility validation before any computation
Feature engineering fails fast if any critical landmark scores below 0.5 visibility. This prevents partial or corrupt feature vectors from reaching the MLP silently.

## 13. Pose orientation not validated by visibility score
MediaPipe visibility scores detect occlusion and out-of-frame landmarks, not
pose angle suitability. A subject at 45 degrees passes visibility checks but
produces systematically corrupted width features. Near-term mitigation:
user-facing instruction to stand facing camera directly, plus a soft warning
if left/right arm lengths diverge by more than 15%. Production solution:
use z coordinates to estimate body rotation angle and reject poses beyond
a threshold.

## 14. _MCDropout subclass instead of model.train() for inference
Relying on callers to set model.train() before MC passes is fragile — any
accidental model.eval() call silently breaks confidence intervals. _MCDropout
subclasses nn.Dropout to always stay active regardless of model mode, making
correct behavior impossible to accidentally bypass.

## 15. weights_only=True on torch.load
PyTorch .pth files are pickle-based and can execute arbitrary code on load.
weights_only=True prevents code execution from malicious or corrupted
checkpoints. Required for any externally-supplied model file.

## 16. RuntimeError at startup not at request time
Missing checkpoint crashes the server at startup with a clear message rather
than failing silently on the first user request. Fail-fast at initialization
is always preferable to silent runtime failures.

## 17. low_confidence flag at std > 5kg threshold
High MC Dropout std indicates epistemic uncertainty — the input is outside
the training distribution, not just a noisy measurement. A 5kg std threshold
(~20kg CI width) is the cutoff where the estimate becomes qualitatively
degraded. The frontend renders this case differently with an explicit warning
rather than just a wider confidence interval bar. The threshold is a
constructor parameter so it can be tuned post-deployment without code changes.

## 18. Bimodal MC distribution problem
If MC passes cluster in two groups, the mean falls in the trough and is the
least likely actual output. Future improvement: check for bimodality in the
50 passes before reporting the mean as the point estimate.

## 19. Checkpoint format includes config metadata
bioscan_model.pth stores model_state_dict, config, and best_val_loss together.
This means any loaded checkpoint is self-documenting — you can inspect what
hyperparameters produced it without needing the original config file.

## 20. Scaler fitted outside BioScanDataset, passed in
The dataset class never fits the scaler itself. Fitting inside the dataset
would make it impossible to guarantee the scaler was fitted on training data
only. The caller (train.py) owns the split and therefore owns the scaler fitting.

## 21. Validation loss intentionally stochastic during training
_MCDropout stays active in eval mode, making validation loss slightly
stochastic. This is acceptable — early stopping and LR scheduling are based
on trends across epochs, not single-epoch precision. The alternative of
disabling MC dropout during training would produce a validation loss that
doesn't reflect inference behavior.

## 22. evaluate.py exits with code 1 if MAE exceeds 6kg
The quality gate is implemented as a sys.exit(1) so it integrates directly
with CI/CD pipelines. A bad model cannot be silently deployed — the pipeline
fails explicitly with a logged reason.

## 23. Scaler co-located with checkpoint, path derived automatically
bioscan_scaler.pkl always lives in the same directory as bioscan_model.pth.
evaluate.py derives the scaler path from the checkpoint path automatically,
preventing accidental model/scaler mismatches which would produce silent
garbage predictions.

## 24. Inference in evaluate.py goes through WeightEstimator not BioScanMLP
Using the full production inference path in evaluation validates the complete
artifact chain — checkpoint loading, config verification, scaler application,
and MC Dropout — not just the model weights in isolation.

## 25. /health endpoint has no model state dependency
Load balancers receive 200 OK even during model loading at startup,
preventing the container from being killed before it's ready to serve
requests.

## 26. Face blur kernel scales with face size
sigmaX = max(w, h) / 4 ensures consistent visual anonymization regardless
of subject distance from camera. Small faces and large close-ups both get
proportionally appropriate blur strength.

## 27. processing_time_ms excludes network I/O
Timer starts at cv2.imdecode not at image.read(). Reports actual
processing time, not upload speed — meaningful for performance monitoring
across different network conditions.

## 28. Canvas unmirrored before submission
Webcam preview is CSS-mirrored (scaleX(-1)) for natural user experience.
Canvas capture applies ctx.scale(-1,1) to undo the mirror before converting
to blob — otherwise MediaPipe receives a horizontally flipped image and
left/right landmarks are swapped.

## 29. facingMode: ideal not exact on mobile
ideal: falls back gracefully to any available camera if rear camera isn't
found. exact: throws a hard error on single-camera devices. ideal: is always
correct for user-facing applications.

## 30. Camera stream stopped after capture on mobile
getTracks().forEach(t => t.stop()) releases the camera after photo capture,
turning off the camera indicator light. Users expect the camera to stop
when they're done — leaving it running is a privacy concern.

## 31. Tests patch at api.main not pipeline.*
Mocks must be applied where the name is used (api.main), not where it
is defined (pipeline.*). Patching at the source leaves already-imported
references unchanged, producing false negatives.

## 32. std_threshold_kg=0.0 used to test low_confidence flag
Setting the threshold to zero guarantees any non-zero MC variance triggers
low_confidence=True without needing to control random seeds or force
specific outputs.
