# Architecture — BioScan AI

This document explains every file in the project in plain English. It is written for a new engineer joining the team who understands Python and machine learning but has not seen this codebase before. The goal is to explain not just what each file does, but why it exists where it does and how it connects to everything around it.

---

## The big picture

BioScan AI estimates a person's body weight from a single photograph. The user submits an image alongside their height (in centimetres) and age through a REST API. The system runs the image through a three-stage pipeline — pose detection, feature engineering, then a neural network — and returns an estimated weight in kilograms along with a 95% confidence interval. There are two ways to submit an image: a desktop browser that uses the computer's webcam, and a mobile phone camera accessed by scanning a QR code. Both paths converge on the same API endpoint.

---

## System Flow

When a user clicks "Capture & Analyse" on the desktop page, `webcam.js` draws the current video frame onto a hidden canvas. Because the live preview is CSS-mirrored (flipped horizontally to feel like a mirror), the canvas capture reverses that flip by applying `ctx.scale(-1, 1)` before drawing — if it did not, MediaPipe would receive a horizontally flipped image and assign left-shoulder coordinates to the right shoulder and vice versa. The canvas is then encoded as a JPEG blob at 92% quality and packed into a `FormData` object along with the user's height and age, and a `fetch` POST fires to `/analyze`. While the request is in flight, the capture button is disabled and a full-screen loading overlay appears so the user cannot trigger a duplicate submission.

On the server, the request arrives at `api/routes/analyze.py`. The route first validates that `height_cm` and `age` are within the accepted ranges, raising an HTTP 422 immediately if they are not. It then reads the uploaded bytes and calls `cv2.imdecode` to produce a BGR numpy array; the processing timer starts at this exact moment so that network upload time is excluded from the reported `processing_time_ms`. Before any pose processing, the route blurs the subject's face using OpenCV's frontal-face Haar cascade. If the cascade finds one or more faces it applies a Gaussian blur to each detected rectangle with a sigma proportional to the face size, so that both close-up and distant faces are rendered unrecognisable. If the cascade finds nothing — which can happen with a profile view or poor lighting — it blurs the top 20% of the frame as a safe fallback, since in a full-body standing photo the head reliably occupies that region.

With the face blurred, the image moves into Stage 1: `PoseDetector.detect()` resizes the image to at most 1280 pixels on its longest side if needed, converts BGR to RGB, acquires a threading lock, and passes the frame to MediaPipe Pose. MediaPipe returns 33 `NormalizedLandmark` objects, each with normalised `x`, `y`, `z` coordinates and a `visibility` confidence score. These become a `PoseDetectionResult`. Stage 2, `compute_features()`, validates that all eight critical landmarks (shoulders, hips, wrists, ankles) have visibility above 0.5, then computes eight body-shape ratios from the landmark coordinates — things like shoulder width relative to hip width, torso height, and a silhouette area proxy. The user's height and age are appended to these eight image-derived values to produce a 10-element `FeatureVector`. Stage 3, `WeightEstimator.predict()`, applies the fitted `StandardScaler` to the feature vector, then runs 50 stochastic forward passes through the MLP with Monte Carlo Dropout active. The 50 outputs are collected into a numpy array; their mean becomes `estimated_weight_kg`, their standard deviation drives the 95% confidence interval bounds (`mean ± 1.96 × std`), and if that standard deviation exceeds 5 kg the result is flagged `low_confidence` to indicate that the input is in a region of the training distribution the model has seen rarely.

The route calculates `processing_time_ms` from the moment of `cv2.imdecode` to the moment `predict()` returns, assembles an `AnalyzeResponse`, and serialises it to JSON. The response travels back to the browser. `webcam.js` unpacks the JSON, writes the estimated weight in large green text, draws the CI bar — a horizontal track spanning the estimate ±2σ with the confidence interval highlighted and a dot marker exactly at the mean — and populates the uncertainty and timing stats below. If `low_confidence` is `True`, a clearly visible amber warning box appears above the stats with the text defined in `api/schemas.py`. The entire round trip from click to result is typically under two seconds on a modern laptop.

The mobile flow is identical from the API's perspective. The only difference is in how the image is captured: `mobile.js` requests the rear camera via `getUserMedia({ facingMode: { ideal: 'environment' } })`, and when the user taps "Take Photo" it captures the frame without mirroring (rear cameras are not mirrored). After capture the live stream is stopped immediately to release the camera indicator light. If `getUserMedia` is unavailable — which can happen in older Android WebViews — the page falls back to a file input with `capture="environment"`, which opens the native camera app instead.

---

## `pipeline/pose_detector.py`

`PoseDetector` is the entry point of the processing pipeline and the only component that touches MediaPipe directly. It is designed to be instantiated once at API startup and reused across every request, because creating a new MediaPipe `Pose` graph on each call carries significant initialisation overhead from loading the TFLite model files.

The `detect()` method accepts either a BGR numpy array (as produced by `cv2.imdecode`) or a file path. When given a numpy array it assumes BGR channel order and converts to RGB, because MediaPipe expects RGB and OpenCV produces BGR by convention. When given a path it calls `cv2.imread`, which also produces BGR, and applies the same conversion. The image is then passed to `_cap_size()`, which resizes it if either dimension exceeds `max_side_px` (default 1280) while preserving the aspect ratio using `cv2.INTER_AREA` interpolation. This cap exists because uncapped uploads can exhaust memory under concurrent load, and 1280 pixels is more than sufficient for MediaPipe's accuracy.

The actual MediaPipe call is protected by a `threading.Lock`. FastAPI serves requests concurrently, and MediaPipe's TFLite interpreter is not documented as thread-safe; without the lock, two requests processed simultaneously can corrupt each other's results silently. The lock serialises calls on a single `PoseDetector` instance; for higher-throughput deployments a pool of instances (one per worker thread) would eliminate the contention entirely.

`detect()` returns a `PoseDetectionResult` dataclass containing the list of 33 raw `NormalizedLandmark` objects, a mean-visibility confidence score computed across all 33 landmarks, and a UTC timestamp recorded just before the inference call. If MediaPipe finds no person in the frame it returns `pose_landmarks = None`, at which point `detect()` raises a `ValueError` with a user-facing message that the route layer converts to an HTTP 422. The class also implements the context manager protocol so callers can use `with PoseDetector() as d:` and be sure `close()` is called even on error.

---

## `pipeline/feature_engineer.py`

`compute_features()` takes a `PoseDetectionResult` from Stage 1 and a `height_cm` and `age` from the API request, and produces a 10-element `FeatureVector` for the MLP.

The first thing the function does is validate visibility. Eight landmarks are considered critical — the left and right shoulders, hips, wrists, and ankles — and if any of them has a `visibility` score below 0.5, the function raises a `ValueError` that names the specific landmark. This fail-fast approach prevents silently corrupt feature vectors from reaching the model; it is far better to return a clean 422 error telling the user to stand with their full body in frame than to return a confident-looking weight estimate derived from partially occluded landmarks.

The actual feature computation uses the `PoseLandmark` enum by name throughout (e.g. `_PL.LEFT_SHOULDER`, not `lm[11]`), which makes the intent of each line unambiguous. All distances are computed in MediaPipe's normalised x/y coordinate space using 2D Euclidean distance; the z coordinate, which is MediaPipe's relative depth estimate rather than a metric value, is excluded because mixing it with the normalised x/y distances introduces noise rather than information.

The eight image-derived features are: `shoulder_width` (Euclidean distance between the two shoulder landmarks), `hip_width` (same for hips), `shoulder_to_hip_ratio` (shoulder_width divided by hip_width, capturing body taper), `torso_height` (distance from the shoulder midpoint to the hip midpoint), `silhouette_area` (shoulder_width multiplied by torso_height, a normalised proxy for cross-sectional area), `left_arm_length` (shoulder to wrist), `right_arm_length` (shoulder to wrist), and `leg_length` (hip midpoint to ankle midpoint). The `height_cm` and `age` values from the API request are appended as the ninth and tenth elements. They are stored as floats, cast explicitly so the vector is uniformly typed. The function also guards against coincident hip landmarks — a hip_width of exactly zero means the subject is sideways-on or the data is corrupt — and raises a `ValueError` in that case before attempting a division.

`FeatureVector` is a plain dataclass. Its `to_list()` method returns the features in the canonical order defined in `training/config.py`, which is the order the scaler and model were trained on. If you ever add or remove a feature, you must update `to_list()`, `Config.feature_columns`, and `model/mlp.py`'s `IN_FEATURES` constant together or inference will silently misalign features with weights.

---

## `pipeline/weight_estimator.py`

`WeightEstimator` is the inference wrapper for Stage 3. Like `PoseDetector`, it is instantiated once at API startup and held in `app.state` for the lifetime of the server process. Construction is deliberately expensive: it loads the `.pth` checkpoint from disk using `torch.load` with `weights_only=True` (which prevents arbitrary code execution from a malicious or corrupted checkpoint file), verifies that the `in_features` value recorded in the checkpoint matches the current `IN_FEATURES` constant, reconstructs a `BioScanMLP`, loads the state dict, and deserialises the `StandardScaler` from the accompanying `.pkl` file using `joblib`. If either file is missing at construction time, a `RuntimeError` is raised immediately with a clear message. The server should not start if its model is absent; failing at startup is far more debuggable than failing on the first user request.

The model is deliberately left in `train()` mode after loading rather than switched to `eval()`. This is not an oversight. `_MCDropout` overrides the standard dropout behaviour to always stay active regardless of the model's mode flag, but keeping the model in `train()` mode makes the intent explicit: this model is intended to be stochastic at inference time, and any future reader should understand that immediately.

`predict()` converts the incoming `FeatureVector` to a float32 numpy array, applies the loaded scaler's `transform()` method, converts to a PyTorch tensor of shape `(1, 10)`, and then runs 50 forward passes inside a `torch.no_grad()` context. `no_grad()` disables gradient tracking for efficiency — it has no effect on dropout, which is controlled by `_MCDropout` independently. Each pass produces a different scalar output because `_MCDropout` drops different neurons on each call. The 50 outputs are collected into a numpy array; their mean is the point estimate, and the standard deviation drives both the 95% CI (`mean ± 1.96 × std`) and the `low_confidence` flag (`std > std_threshold_kg`, default 5 kg). A high standard deviation means the 50 passes disagreed substantially, which indicates the input feature vector lies in a region of the training distribution the model has seen infrequently — the estimate may not be reliable regardless of how plausible the number looks. The result is returned as a `WeightEstimationResult` dataclass.

---

## `model/mlp.py`

This file defines the neural network architecture and one critical supporting class, `_MCDropout`. Understanding why `_MCDropout` exists is essential to understanding the system's uncertainty estimation.

Standard PyTorch `nn.Dropout` checks the module's `training` flag on every forward pass and becomes a no-op when the model is in `eval()` mode. That is the correct behaviour for most applications — you train with dropout for regularisation and disable it at inference for deterministic outputs. But BioScan AI needs stochastic inference: 50 forward passes through the same input should produce 50 different outputs, and their spread is interpreted as the model's epistemic uncertainty. `_MCDropout` is a two-line subclass of `nn.Dropout` that overrides `forward()` to always call `F.dropout(..., training=True)`, bypassing the mode check entirely. This makes it impossible for calling code to accidentally disable dropout at inference time, even if someone calls `model.eval()` out of habit or convention.

`BioScanMLP` is a four-layer network. The input layer accepts 10 features (the length of `FeatureVector.to_list()`). There are three hidden layers of widths 128, 64, and 32 neurons, each followed by a ReLU activation and an `_MCDropout` layer with dropout probability 0.3. The final output layer is a single neuron with no activation, producing a raw scalar that is interpreted as the estimated weight in kilograms. The `forward()` method passes the input through `nn.Sequential` and applies `squeeze(-1)` to the output to collapse the trailing dimension, so that a batch of size `n` produces a tensor of shape `(n,)` rather than `(n, 1)`. `IN_FEATURES = 10` is defined as a module-level constant so that both `FeatureVector.to_list()` and `WeightEstimator.__init__` can import and assert against it, keeping the feature count contract explicit and machine-checkable.

---

## `model/dataset.py`

`BioScanDataset` wraps a pandas DataFrame of anthropometric features and weight labels into a PyTorch `Dataset` for use with `DataLoader`. It takes a fitted scaler, a list of feature column names, and a target column name at construction time. Critically, it calls `scaler.transform()` — never `fit()` — to scale the features. The scaler must have been fitted on the training split only and passed in from outside; the dataset class deliberately has no way to fit a scaler itself, which enforces the rule that validation statistics must never contaminate the scaling transform.

All scaling and tensor conversion is done in `__init__` rather than in `__getitem__`. This means the preprocessing cost is paid once when the dataset is constructed rather than once per sample per epoch, which is a substantial speedup when the dataset is large. `__getitem__` simply indexes into two pre-built tensors, `_X` and `_y`, and returns a `(feature_tensor, weight_tensor)` pair.

---

## `model/trainer.py`

`Trainer` owns the training loop. It takes a `Config`, a `BioScanMLP`, a training `DataLoader`, a validation `DataLoader`, and the fitted scaler. It does not instantiate any of these — that is `train.py`'s job. The separation means the training loop itself is testable without touching the file system or the dataset loading machinery.

The loop runs for up to `config.epochs` epochs. After each epoch, a `ReduceLROnPlateau` scheduler halves the learning rate if validation loss has not improved for 10 consecutive epochs. If validation loss has not improved for 20 consecutive epochs, training stops early and the best weights seen so far are restored before returning. At the end of training, the trainer saves two artefacts to `model/checkpoints/`: `bioscan_model.pth`, which contains the model state dict, a config snapshot, and the best validation loss; and `bioscan_scaler.pkl`, the fitted scaler serialised with `joblib`. Both must always be deployed together.

---

## `training/config.py`

`Config` is a single dataclass that defines every hyperparameter, path, and structural constant for a training run. The most important field is `feature_columns`, an ordered list of strings that defines the canonical feature vector layout. This list must match `FeatureVector.to_list()` in `pipeline/feature_engineer.py` exactly — if the order differs, features will be silently mapped to the wrong model weights at inference time. `Config.__post_init__` validates that `len(feature_columns) == in_features` at construction time to catch the most common mistake early. All other training scripts import from `Config` rather than defining their own constants.

---

## `training/train.py`

This is the script you run to train the model: `python training/train.py`. It instantiates a `Config`, loads the ANSUR II and SMPL CSVs with `pandas.read_csv`, validates that the required columns are present in both, concatenates the two DataFrames with a `source` column to preserve traceability, performs an 80/20 train/validation split using `sklearn.train_test_split`, fits a `StandardScaler` on the training features only, constructs `BioScanDataset` and `DataLoader` instances for both splits, instantiates `BioScanMLP` with the config's `in_features` and `dropout_p`, hands everything to `Trainer.train()`, and exits. The script is intentionally a thin orchestration layer; all the interesting logic is in the files it calls.

---

## `training/evaluate.py`

`evaluate.py` measures a trained checkpoint's prediction quality against a held-out test CSV. It loads a `WeightEstimator` (not a raw `BioScanMLP`) so that the evaluation runs through the exact same inference path — scaler application, MC Dropout, confidence interval computation — that production uses. For each row in the CSV it constructs a `FeatureVector` from the feature columns and calls `estimator.predict()`, then compares the result to the ground-truth `weight_kg` column. It computes MAE, RMSE, and CI calibration, where CI calibration is the fraction of ground-truth weights that fall within the predicted 95% confidence interval (a well-calibrated model should score near 0.95). If MAE exceeds 6 kg, the script exits with code 1, which causes any CI/CD pipeline running it as a quality gate to fail explicitly rather than silently deploying a bad model.

---

## `training/data/raw/`

This directory holds the original, unmodified source data: ANSUR II CSV files and SMPL synthetic renders as they were downloaded or generated. Nothing in the codebase writes to this directory. It is gitignored because the files are large and in some cases subject to data use agreements. A new team member needs to obtain and place the data here manually before running training.

---

## `training/data/processed/`

This directory holds pre-extracted feature vectors stored as numpy arrays. Running the full MediaPipe pipeline on every SMPL image during every training run would be extremely slow, so a preprocessing step runs Stage 1 and Stage 2 over all images once and caches the resulting feature vectors here. `dataset.py` loads from this cache during training. Like `raw/`, this directory is gitignored.

---

## `model/checkpoints/`

This directory stores the `.pth` checkpoint and the `.pkl` scaler file produced at the end of a training run. Both artefacts must always be deployed together. `WeightEstimator.__init__` enforces this at startup: it checks for both files and raises `RuntimeError` immediately if either is missing, rather than failing on the first request. The directory is gitignored because checkpoint files are large binary artefacts that belong in dedicated model storage rather than version control.

---

## `api/main.py`

This is the FastAPI application object — the thing `uvicorn` loads when you start the server. The most important design choice here is the lifespan context manager. When the server starts, `lifespan` instantiates both `PoseDetector` and `WeightEstimator` exactly once and stores them in `app.state`. Both objects are designed to be created once and reused: `PoseDetector` holds an open MediaPipe graph and `WeightEstimator` holds a loaded PyTorch model and scaler in memory. Creating either per request would add hundreds of milliseconds of initialisation overhead to every call. The file paths for the checkpoint and scaler are resolved to absolute paths at import time using `Path(__file__).resolve()` so that the server starts correctly regardless of which directory `uvicorn` is invoked from.

The `app` also registers CORS middleware with `allow_origins=["*"]` — correct for local development but explicitly commented as requiring restriction before production deployment — mounts the `frontend/` directory under `/static` so the HTML, JavaScript, and QR code image are all served by the same process, and defines a `/health` endpoint that returns `{"status": "ok"}`. The health endpoint has no dependency on `app.state`, which means load balancers receive a 200 response immediately when the server process starts, before model loading completes, preventing the container from being killed prematurely during startup.

---

## `api/routes/analyze.py`

This file is the implementation of `POST /analyze`, the only user-facing route in the system. It is worth understanding in some detail because it is where all the pipeline stages meet.

The Haar cascade classifier is loaded once at module import time into `_FACE_CASCADE`. This is not inside the route function or the lifespan — it is a module-level constant. `cv2.CascadeClassifier` reads an XML file and initialises quickly, so loading it at import time adds negligible startup cost while ensuring it is never reloaded per request.

The `_blur_face()` helper converts the image to grayscale, runs `detectMultiScale` with `scaleFactor=1.1`, `minNeighbors=5`, and a minimum face size of 30×30 pixels, and blurs each detected face rectangle with `cv2.GaussianBlur` using a sigma of `max(face_width, face_height) / 4`. This sigma scaling is deliberate: a face that occupies 400×400 pixels in a close-up photo needs a much stronger blur than a 60×60-pixel face in a wide shot. If no face is detected — which happens most often with profile views or partial occlusion — the fallback blurs the top 20% of the frame. In a properly framed full-body standing photo, the head always sits in this region, so the fallback provides meaningful protection even when face detection fails.

The `analyze()` route function validates `height_cm` and `age` manually rather than relying solely on Pydantic, because the form fields come in as raw strings from a multipart upload and we want to return our own clear error messages. After validation it reads the image bytes, starts the processing timer at `cv2.imdecode`, blurs the face, and runs Stages 1, 2, and 3 in sequence, accessing the shared `PoseDetector` and `WeightEstimator` instances via `request.app.state`. Both `ValueError` exceptions — raised by `PoseDetector` when no person is found and by `compute_features` when landmarks are too occluded — are caught and returned as HTTP 422 responses with the original message text, which is already written to be user-readable. A bare `except Exception` catches all unexpected failures, logs the full traceback server-side with `_log.exception`, and returns a generic 500 message to the client that does not leak internal details.

---

## `api/schemas.py`

This file defines three things: the `AnalyzeRequest` Pydantic model used for documentation and reuse, the `AnalyzeResponse` Pydantic model that FastAPI uses to serialise the endpoint's return value, and `LowConfidenceWarning`, a module-level string constant containing the exact user-facing message that should appear when `AnalyzeResponse.low_confidence` is `True`. Defining the warning text here ensures that both the API (which documents it in the OpenAPI spec) and the frontend JavaScript (which copies it verbatim) stay in sync — the string is defined once and referenced everywhere.

`AnalyzeResponse` includes `input_height_cm`, which is the user-submitted height passed through to the response unchanged. This field exists to make explicit that height is user input, not a computed value. The name itself is the documentation.

---

## `frontend/index.html`

This is the desktop user interface. It uses a dark theme with CSS custom properties throughout, styled for a professional medical or technical context. The layout is a single centred column with a max-width of 700 pixels. At the top, two number inputs collect `height_cm` and `age`. Below them, a 16:9 aspect-ratio container holds the webcam `<video>` element with CSS `transform: scaleX(-1)` applied to mirror the preview — a standard selfie-camera convention. Below the video are two buttons: "Capture & Analyse" (which drives the primary flow) and "Switch to Mobile Camera" (which toggles a section containing the pre-generated QR code image). The results section below is hidden until a response arrives; it shows the estimated weight in a large green number, a CI bar visualisation, a stats grid for uncertainty and processing time, and — conditionally — the `LowConfidenceWarning` in an amber box. A fixed-position loading overlay with a CSS-animated spinner covers the page while the API call is in flight. All JavaScript is in `webcam.js`; this file contains no inline scripts.

---

## `frontend/static/js/webcam.js`

This script is wrapped in an IIFE (immediately invoked function expression) to avoid polluting the global scope. It follows a straightforward lifecycle: initialise the camera on page load, capture on button click, POST to the API, render the result.

`initWebcam()` calls `navigator.mediaDevices.getUserMedia` requesting `facingMode: 'user'` at 1280×720 ideal resolution. If the user denies permission, the error handler disables the capture button and shows a message inside the video container explaining how to fix it.

`captureFrame()` is where the mirror reversal happens. It sets the canvas to the video's native dimensions, then applies `ctx.translate(canvas.width, 0)` followed by `ctx.scale(-1, 1)` before drawing the video. This sequence flips the canvas horizontally, undoing the CSS mirror so that the JPEG sent to the server is a geometrically correct image. Without this step, MediaPipe would receive a mirrored image and assign left-shoulder coordinates to the anatomical right shoulder. The canvas is then encoded to a JPEG blob at 92% quality via `canvas.toBlob()`.

`submit()` validates the input fields client-side — both to give instant feedback without a round trip and to avoid sending a request the server will reject — then assembles a `FormData` with the blob, `height_cm`, and `age`, and calls `fetch('/analyze', { method: 'POST', body: fd })`. It sets the loading state before the fetch and clears it in a `.catch()` / `.then()` chain regardless of outcome, so the button is always re-enabled even on network error.

`renderResults()` populates all result elements and calls `renderCIBar()`, which calculates a dynamic display range of `[max(0, ci_low − 2σ), ci_high + 2σ]` with a minimum 5 kg padding on each side. This ensures that a very narrow confidence interval does not produce a bar that is 100% filled, and that a very wide interval does not overflow the track. The fill element's `left` and `width` CSS properties are set as percentages of the track width; the marker dot is positioned at the estimate, which is always the midpoint of the fill because the CI is symmetric by construction.

---

## `frontend/mobile.html`

This is the mobile-optimised capture page opened when a user scans the QR code. It is served at `/static/mobile.html`. The layout is a single narrow column appropriate for phone screens, with a 3:4 aspect-ratio camera container (portrait orientation, better suited to full-body shots than 16:9). The flow has two phases: first the camera is live and the user taps "Take Photo"; after capture, the live video is replaced by a preview of the still image and the user chooses between "Analyse" and "Retake" before submitting. This explicit preview step is more forgiving on mobile than the desktop's immediate capture, since phone camera framing is harder and users frequently need to retake shots. Like `index.html`, all JavaScript is external. The page includes `maximum-scale=1.0` in the viewport meta tag to prevent iOS Safari from zooming into input fields, which would break the layout.

---

## `frontend/static/js/mobile.js`

Like `webcam.js`, this is an IIFE. The camera initialisation is deliberately tolerant: `getUserMedia` is called with `facingMode: { ideal: 'environment' }` rather than `{ exact: 'environment' }`. Using `ideal` means the browser will use the rear camera if available but fall back to any available camera rather than throwing a hard error on devices with only one camera. If `getUserMedia` itself is unavailable — which covers older Android WebViews and some in-app browsers — `enableFileInputFallback()` hides the video element and shows a `<label>` styled as a button that wraps a `<input type="file" capture="environment">`. On mobile, this input invokes the native camera app and returns the captured photo as a `File` object.

When the user taps "Take Photo", `captureFromStream()` draws the current video frame to the canvas without any horizontal flip (rear cameras produce geometrically correct images) and calls `canvas.toBlob()` to produce a JPEG. Before showing the preview, `showPreview()` stops all tracks on the active media stream by calling `getTracks().forEach(t => t.stop())`. This is important on mobile: a running camera stream keeps the camera indicator light on, drains the battery, and implies to the user that they are still being recorded. Stopping the tracks as soon as the photo is taken is both a privacy courtesy and expected device behaviour.

`submit()` handles both capture paths uniformly. `_capturedBlob` holds either a `Blob` from the canvas path or a `File` from the file input path. Both are valid `FormData` values for the same `image` field, so the fetch call is identical regardless of which path was used. The `renderResults()` and `renderCIBar()` functions are identical in logic to their counterparts in `webcam.js`.

---

## `frontend/static/css/`

This directory is reserved for stylesheets shared between the desktop and mobile pages. Currently both pages use embedded `<style>` blocks. Moving shared styles here when the CSS grows large enough to justify the separation is the expected next step.

---

## `frontend/qr_generator.py`

This is a developer utility, not part of the request pipeline. Run it with `python frontend/qr_generator.py <ngrok-url>` whenever the server address changes. It constructs the mobile page URL as `{base_url}/static/mobile.html`, creates a `QRCode` object with `ERROR_CORRECT_M` (15% error recovery, chosen over `ERROR_CORRECT_L` because it scans more reliably on small phone screens at the cost of a slightly denser grid), and saves the resulting PNG to `frontend/static/qr_mobile.png`. The desktop UI loads that image from `./static/qr_mobile.png` and displays it in the "Switch to Mobile Camera" section. The URL is also printed to stdout so the developer can verify what was encoded. If the `qrcode` package is not installed, the script exits with a clear install instruction rather than a raw `ImportError`.

---

## `tests/test_pipeline.py`

This file tests Stages 1 and 2 of the pipeline. The key design decision is that `mediapipe.solutions.pose.Pose` is patched at the `mediapipe` module level before `PoseDetector` is instantiated in each test, which prevents TFLite model files from being loaded during the test run. Each `pose_detector` fixture yields a `(PoseDetector, mock_pose_instance)` tuple so that individual tests can configure `mock_pose_instance.process.return_value` freely without affecting other tests.

The `compute_features` tests bypass `PoseDetector` entirely and construct `PoseDetectionResult` objects directly using `types.SimpleNamespace` as a stand-in for `NormalizedLandmark`. `SimpleNamespace` works because `compute_features` only accesses `.x`, `.y`, and `.visibility` attributes on each landmark, which `SimpleNamespace` satisfies cheaply. The key landmarks are placed at geometrically clean positions — shoulder width of exactly 0.30, hip width of exactly 0.20 — so that expected values are easy to compute by hand and assert precisely.

---

## `tests/test_model.py`

This file tests `BioScanMLP` and `WeightEstimator`. The `BioScanMLP` tests are straightforward tensor shape checks plus the critical MC Dropout test: `model.eval()` is called explicitly, then 100 forward passes are run on the same input inside `torch.no_grad()`, and the standard deviation of the outputs is asserted to be greater than zero. If `_MCDropout` were broken — if it were behaving like standard `nn.Dropout` and becoming a no-op in eval mode — all 100 outputs would be identical and this assertion would fail.

The `WeightEstimator` tests use a `saved_checkpoint` fixture that writes a real checkpoint to `tmp_path`: it creates a `BioScanMLP`, calls `torch.save` with the full expected checkpoint dict structure including config metadata, fits a `StandardScaler` on random data, and saves it with `joblib`. This exercises the full `WeightEstimator.__init__` path — file existence checks, `torch.load` with `weights_only=True`, `in_features` verification, state dict loading, and scaler deserialisation — with weights that are random but structurally correct. The `test_low_confidence_flag` test instantiates a `WeightEstimator` with `std_threshold_kg=0.0`, which guarantees `low_confidence=True` for any non-zero MC standard deviation. Because `_MCDropout` is always active, the 50 passes will always produce different outputs, so the standard deviation will always be greater than zero.

---

## `tests/test_api.py`

This file tests the full HTTP layer using FastAPI's `TestClient`. The key challenge is that the app's lifespan tries to load model files from disk when `TestClient(app)` is entered. The fixture solves this by patching both `api.main.PoseDetector` and `api.main.WeightEstimator` before the lifespan runs. Patching at `api.main` rather than at `pipeline.pose_detector` or `pipeline.weight_estimator` is correct because Python's `patch` replaces the name in the namespace where it is used — `api.main` imports both classes, so `api.main.PoseDetector` is the reference the lifespan actually calls. After the patch, the lifespan stores mock instances in `app.state.pose_detector` and `app.state.weight_estimator` exactly as it does in production, so the route accesses them identically.

`compute_features` runs for real between the two mocked stages in the happy-path tests, which means the `PoseDetectionResult` returned by the mock detector must contain valid geometry — visible landmarks at distinct positions with non-zero hip width. The `_make_pose_result()` helper constructs this with eight critical landmarks at sensible coordinates. The "no person detected" test sets `side_effect = ValueError(...)` on the mock detector after fixture setup and resets it in a `finally` block, though since the fixture is function-scoped each test gets its own mock anyway.

---

## `requirements.txt`

All Python dependencies are pinned to specific versions compatible with Python 3.11. NumPy is pinned at 1.26.4 (the last 1.x release) rather than 2.x because both `torch==2.3.0` and `mediapipe==0.10.11` have known compatibility issues with NumPy 2.0. `python-multipart` is included even though it is never directly imported: FastAPI raises a runtime error on the first `Form` or `UploadFile` request if it is absent. `httpx` is included because `fastapi.testclient.TestClient` raises `ImportError` at construction time without it, even though no test file imports `httpx` directly. `Pillow` is pinned explicitly because both `qrcode[pil]` and `mediapipe` depend on it and pinning it prevents pip from silently resolving conflicting versions between the two.

---

## `DECISIONS.md`

This file records every significant architectural decision made during design and implementation, along with the reasoning behind it. It covers choices like why an MLP is used instead of a CNN, why Monte Carlo Dropout was chosen over deep ensembles, why `input_height_cm` is a passthrough rather than an estimated value, why the Haar cascade falls back to blurring the top 20% of the frame, and why tests patch at `api.main` rather than at the source module. Without this file, a new engineer encountering an unusual design choice has no way to discover whether it was deliberate, why it was made, and what alternatives were considered. It should be updated whenever a significant design choice is made or revised.

---

## `CLAUDE.md`

This file is read by Claude Code at the start of every session. It contains the commands needed to run, test, and train the system, plus a compressed summary of the architecture focused on the facts that are hardest to infer from reading individual files — the data flow between pipeline stages, the constraint that the scaler must be saved alongside the checkpoint, the HTTPS requirement for mobile camera access, and the canonical feature order. Think of it as the minimum context an AI assistant needs to work effectively on this codebase without reading every file from scratch.
