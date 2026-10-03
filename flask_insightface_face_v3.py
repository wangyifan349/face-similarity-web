"""Face Similarity Web (InsightFace).

On startup it asks which folder holds your face photos, then brings the web
service up: open the page, pick one query image, then pick a set of candidate
images, and you see how much each candidate resembles the query; or let the
query image be compared against your face folder. Faces are detected by SCRFD
and described by ArcFace, every face becomes a 512-d descriptor (L2
normalized), and the score is the cosine similarity of two face descriptors
times 100. Different photos of the same person measure 77-79 here, different
people -3 to +1. This program does not decide for you whether two faces are
"the same person": calibrate that threshold on your own samples.

Scores come from ArcFace, which is not the same model as the dlib version
(flask_face_match_v2.py), so the numbers cannot be compared across the two. The
page layout, the colors and the fixed 0-100 similarity bar are the same as in
the dlib version, with two differences: there is no detection mode picker
(SCRFD is the only detector), and the results carry no detector field.

Install the dependencies (Python 3.9 or newer):

    pip install flask opencv-python numpy insightface onnxruntime

To use an NVIDIA GPU, swap onnxruntime for onnxruntime-gpu.

Run it:

    python flask_insightface_face_v3.py

Then type the face library folder when prompted; pressing Enter uses
face_library next to this script. Images may be .jpg .jpeg .png .bmp .webp and
may live in subfolders. The page neither shows nor accepts a folder path, so
only you know where it is.

The buffalo_l models (about 300 MB) are downloaded automatically on first run
into ~/.insightface/models/buffalo_l; when insightface_models/buffalo_l or
models/buffalo_l already sits next to this script they are used in place, with
no network access.

Every setting is either in the code or asked for at runtime; no environment
variable is ever read.
"""

import socket
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime
from flask import Flask, jsonify, render_template_string, request

# Only the detection and recognition models of buffalo_l are loaded. The
# landmarks and the gender/age models are never used, and loading them would
# only slow startup.
ALLOWED_MODULES = ["detection", "recognition"]
REQUIRED_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
MODEL_FOLDER = Path("~/.insightface").expanduser() / "models" / "buffalo_l"
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# The InsightFace landmarks model calls a deprecated scikit-image interface
# and warns once per detected face. Similarity scoring does not need that
# notice, so it is switched off once and for all here.
warnings.filterwarnings("ignore", message=r".*estimate.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.*")

# The face library folder, filled in by the question asked at startup; it is
# used from then on.
LIBRARY_FOLDER = SCRIPT_DIRECTORY / "face_library"

# The already loaded FaceAnalysis. Building it takes more than two seconds, so
# it is built once and every request then uses it.
FACE_APP = None
MODEL_SUMMARY = ""

# folder -> (modified time, the descriptor of every face in that file)
LIBRARY_CACHE = {}
LIBRARY_LOCK = threading.Lock()


# =============================================================================
# Finding and loading the models
# =============================================================================


def find_model_directory() -> Optional[Path]:
    """A buffalo_l folder that is already on disk, or None when there is none.

    Next to this script wins, so shipping the models beside the program makes
    it run offline; only when there is really nothing is InsightFace allowed
    to download.
    """
    places = [
        SCRIPT_DIRECTORY / "insightface_models" / "buffalo_l",
        SCRIPT_DIRECTORY / "models" / "buffalo_l",
        SCRIPT_DIRECTORY / "buffalo_l",
        MODEL_FOLDER,
    ]
    for place in places:
        if not place.is_dir():
            continue
        complete = True
        for file_name in REQUIRED_MODEL_FILES:
            if not (place / file_name).is_file():
                complete = False
                break
        if complete:
            return place
    return None


def load_model() -> str:
    """Load the detection and recognition models, return a one line log summary.

    Runs once per process; every later request uses FACE_APP directly.
    """
    global FACE_APP
    global MODEL_SUMMARY

    from insightface.app import FaceAnalysis

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    found = find_model_directory()
    if found is None:
        # Let InsightFace download into MODEL_FOLDER itself.
        FACE_APP = FaceAnalysis(
            name="buffalo_l",
            root=str(MODEL_FOLDER.parents[1]),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = f"{MODEL_FOLDER} (downloaded this run)"
    else:
        FACE_APP = FaceAnalysis(
            name=str(found),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = str(found)

    # ctx_id only means something with the CUDA backend.
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    FACE_APP.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)

    MODEL_SUMMARY = (
        f"model=buffalo_l dim=512 providers={','.join(providers)} models={where}"
    )
    return MODEL_SUMMARY

def get_app():
    """The already loaded models, loading them first if needed."""
    if FACE_APP is None:
        load_model()
    return FACE_APP


# =============================================================================
# Turning every kind of input into a BGR array
#
# Everything below this point accepts one shape only: a contiguous BGR uint8
# array, which is what both OpenCV and InsightFace want.
# =============================================================================


ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def to_bgr_array(image: ImageInput, input_is_bgr: bool = True) -> np.ndarray:
    """Normalize a path, a blob of bytes or an array into a contiguous BGR
    uint8 array.

    input_is_bgr=True means the array handed in is already in BGR order (that
    is OpenCV's own order, hence the default); False means it is RGB, and this
    function converts it to BGR. The flag has no effect for a path or bytes.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            array = np.repeat(image[:, :, None], 3, axis=2)      # gray to 3 channels
        elif input_is_bgr:
            array = image
        else:
            array = image[:, :, ::-1]                              # RGB to BGR
        if array.dtype != np.uint8:
            # The other common convention is a float array in [0, 1].
            if np.issubdtype(array.dtype, np.floating) and array.size > 0:
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype("uint8")
        return np.ascontiguousarray(array)

    # cv2.imread cannot open a non-ASCII path on Windows, so the bytes are read
    # here and decoded below; every kind of input goes through this one path.
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        data = image_path.read_bytes()

    decoded = cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Image cannot be decoded; it may be corrupted or not an image")
    return np.ascontiguousarray(decoded)


# =============================================================================
# Descriptors and scores
#
# This block is the whole engine. The web part only carries images and JSON
# around and does nothing else.
# =============================================================================


def _face_box_area(face) -> float:
    """The area of a detection box, in pixels."""
    x1 = float(face.bbox[0])
    y1 = float(face.bbox[1])
    x2 = float(face.bbox[2])
    y2 = float(face.bbox[3])
    width = x2 - x1
    height = y2 - y1
    if width < 0.0 or height < 0.0:
        return 0.0
    return width * height


def _l2_normalized(vector) -> Optional[np.ndarray]:
    """One descriptor divided by its own length. An all zero vector gives None."""
    array = np.asarray(vector, dtype="float32").reshape(-1)
    length = float(np.linalg.norm(array))
    if length == 0.0:
        return None
    return array / length


def encode_faces(image: ImageInput, input_is_bgr: bool = True) -> list:
    """One 512-d descriptor per face in the image, already L2 normalized.

    Sorted by detection box area from large to small, so element 0 is the
    biggest face in the photo and the caller never has to care how many faces
    there are. Returns an empty list when no face is detected.
    """
    detected = encode_faces_with_scores(image, input_is_bgr)
    embeddings = []
    for face in detected:
        embeddings.append(face["embedding"])
    return embeddings


def encode_faces_with_scores(
    image: ImageInput, input_is_bgr: bool = True
) -> list:
    """Like encode_faces, plus the detector's own confidence and box area.

    Each item is {"embedding": descriptor, "det_score": detection confidence,
    "area": box area}. Only needed when you want to know how sure the detector
    was.
    """
    faces = get_app().get(to_bgr_array(image, input_is_bgr))

    results = []
    for face in faces:
        # `or []` cannot be used here: a descriptor is a NumPy array, and
        # testing an array for truth raises "truth value of an array with more
        # than one element is ambiguous".
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            continue
        vector = _l2_normalized(embedding)
        if vector is None:
            continue
        results.append(
            {
                "embedding": vector,
                "det_score": float(getattr(face, "det_score", 0.0)),
                "area": _face_box_area(face),
            }
        )

    # Biggest first, so the caller can take element 0 as *the* face of a photo.
    results.sort(key=lambda row: row["area"], reverse=True)
    return results


def cosine_similarity_percent(first: np.ndarray, second: np.ndarray) -> float:
    """The cosine similarity of two descriptors, expressed as a percentage."""
    left = np.asarray(first, dtype="float32").reshape(-1)
    right = np.asarray(second, dtype="float32").reshape(-1)
    if left.size != right.size:
        raise ValueError(f"Descriptor sizes differ: {left.size} and {right.size}")
    return float(np.dot(left, right)) * 100.0


def best_pair_percent(first_faces: Sequence, second_faces: Sequence,
                      largest_only: bool = False) -> float:
    """The highest score between two photos.

    Group photos line up on their own, no matter how many faces each side
    holds. With largest_only=True only the biggest face of each side is
    compared.
    """
    if largest_only:
        return cosine_similarity_percent(first_faces[0], second_faces[0])

    highest = -100.0
    for first in first_faces:
        for second in second_faces:
            score = cosine_similarity_percent(first, second)
            if score > highest:
                highest = score
    return highest


def best_scores_percent(query_faces: Sequence, candidate_faces: Sequence) -> list:
    """One best score per candidate, in the candidates' original order.

    The query image may hold several faces (a group photo), so every candidate
    is scored against the query face that fits it best. It is all done with a
    single matrix multiplication rather than nested Python loops, which makes
    a clear difference on a large library.
    """
    query_matrix = np.vstack(query_faces).astype("float32")
    candidate_matrix = np.vstack(candidate_faces).astype("float32")
    return (candidate_matrix @ query_matrix.T).max(axis=1) * 100.0


def face_similarity_percent(
    image_a: ImageInput,
    image_b: ImageInput,
    input_is_bgr: bool = True,
    largest_only: bool = False,
) -> Optional[float]:
    """Face similarity between two images, as a percentage.

    See to_bgr_array for what input_is_bgr means. With largest_only=True only
    the biggest face of each image is compared; by default every pair of faces
    is compared and the highest score wins, so a group photo needs no cropping
    first.

    Returns None when no face is detected in one of the images. A higher score
    means more alike; different photos of the same person measure 77-79 here,
    different people -3 to +1.
    """
    faces_a = encode_faces(image_a, input_is_bgr)
    faces_b = encode_faces(image_b, input_is_bgr)
    if not faces_a or not faces_b:
        return None

    score = best_pair_percent(faces_a, faces_b, largest_only)
    # Floating point drift can push the result slightly out of range; clamp it
    # back into -100 to 100.
    if score > 100.0:
        return 100.0
    if score < -100.0:
        return -100.0
    return score


# =============================================================================
# The face library
#
# The face library is simply any folder full of images. Encoding a whole
# library takes a while, so every file is encoded once and cached, with the
# file's modified time deciding whether it is still valid: edit, add or delete
# an image and the next comparison shows it.
# =============================================================================


def library_images() -> list:
    """Every image in the face library folder, subfolders included, in a
    stable order."""
    found = []
    if not LIBRARY_FOLDER.is_dir():
        return found
    for path in LIBRARY_FOLDER.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    found.sort()
    return found


def _embeddings_of(path: Path) -> list:
    """The descriptors of every face in one file, taken from cache when the
    file has not changed.

    A file that cannot be read, or that holds no face, gives an empty list
    rather than an error: one bad photo must not take the whole library down.
    """
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        return []          # deleted between the folder scan and now

    key = str(path)
    cached = LIBRARY_CACHE.get(key)
    if cached is not None and cached[0] == modified_time:
        return cached[1]

    faces = []
    try:
        faces = encode_faces(path)
    except (OSError, ValueError):
        faces = []
    LIBRARY_CACHE[key] = (modified_time, faces)
    return faces


def load_library() -> tuple:
    """Every face in the library, returned as (list of labels, descriptor
    matrix).

    A label is the file name; when one photo holds several faces " #2" and
    " #3" are appended. The matrix has one row per label, in the same order.
    """
    images = library_images()
    labels = []
    embeddings = []
    alive = []

    # One library scan at a time: the ONNX session and the detection network
    # both suffer when two requests crowd them.
    with LIBRARY_LOCK:
        for index, path in enumerate(images, start=1):
            if len(images) >= 20:
                print(f"  Loading library {index}/{len(images)}  {path.name}", flush=True)

            alive.append(str(path))
            position = 0
            for embedding in _embeddings_of(path):
                position += 1
                if position == 1:
                    labels.append(path.name)
                else:
                    labels.append(f"{path.name} #{position}")
                embeddings.append(embedding)

        # Images deleted from the library have their cache entries dropped too.
        for key in list(LIBRARY_CACHE):
            if key not in alive:
                del LIBRARY_CACHE[key]

    if not embeddings:
        return labels, np.zeros((0, 512), dtype="float32")
    return labels, np.vstack(embeddings).astype("float32")


# =============================================================================
# Turning scores into a result table
# =============================================================================


def rank_matches(
    query_faces: list,
    labels: list,
    candidate_faces: Sequence,
    group_label: str,
    top_results: int,
) -> list:
    """Score every candidate against the query image and keep only the top few.

    labels holds the display name of each candidate, in the same order as
    candidate_faces. group_label says where the candidates came from and is
    shown as the "kind" of every row ("Uploaded candidate" for uploads,
    "Library candidate" for the face library).

    The rows returned are {"rank", "kind", "candidate", "similarity"}, with
    "best": True added to the first one. An empty list comes back when there
    are no candidates.
    """
    scores = best_scores_percent(query_faces, candidate_faces)
    order = np.argsort(-scores)
    count = min(len(scores), max(1, int(top_results)))

    matches = []
    for position in range(count):
        index = int(order[position])
        matches.append(
            {
                "rank": position + 1,
                "kind": group_label,
                "candidate": labels[index],
                "similarity": round(float(scores[index]), 2),
            }
        )
    if matches:
        matches[0]["best"] = True
    return matches


def disambiguate(names: list) -> list:
    """Number files that share a name, so no two rows of the result table look
    exactly alike."""
    totals = {}
    for name in names:
        totals[name] = totals.get(name, 0) + 1

    labels = []
    seen = {}
    for name in names:
        if totals[name] == 1:
            labels.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        labels.append(f"{name} ({seen[name]})")
    return labels

# =============================================================================
# The web page
# =============================================================================


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


class RequestError(Exception):
    """Something about this request is wrong; the message is shown to the user."""


def error_response(message: str, status: int = 400):
    """The one error body both the page and a person reading the API see."""
    return jsonify({"ok": False, "error": message}), status


@app.errorhandler(RequestError)
def handle_request_error(error: RequestError):
    return error_response(str(error))


@app.errorhandler(413)
def handle_too_large(_error):
    return error_response("Uploaded file is too large, the limit per request is 64 MB", 413)


def read_uploads(field_name: str) -> list:
    """Every uploaded file of a field, as (filename, bytes) pairs."""
    uploads = []
    for item in request.files.getlist(field_name):
        if item and item.filename:
            uploads.append((item.filename, item.read()))
    return uploads


def encode_upload(data: bytes, label: str, file_name: str) -> list:
    """The descriptors of every face in one uploaded image.

    An unusable image (decode failure, corrupted, no face inside) raises
    RequestError right away; the message names the offending file, and label
    is "Query image" or "Candidate image".
    """
    try:
        faces = encode_faces(data)
    except (OSError, ValueError) as error:
        raise RequestError(f"{label} could not be processed: {error}: {file_name}")
    if not faces:
        raise RequestError(f"No face detected in {label.lower()}: {file_name}")
    return faces


def query_from_request() -> tuple:
    """The query image of this request, as (filename, descriptors of every
    face)."""
    uploads = read_uploads("query")
    if not uploads:
        raise RequestError("Please choose one query image")
    if len(uploads) > 1:
        raise RequestError("Only one query image may be selected")

    name, data = uploads[0]
    return name, encode_upload(data, "Query image", name)


def top_results_from_request() -> int:
    """How many rows to return; anything out of range is clamped."""
    raw_value = request.form.get("top_results") or request.args.get("top_results") or ""
    text = str(raw_value).strip()
    if not text.isdigit():
        return 10
    wanted = int(text)
    if wanted < 1:
        return 1
    if wanted > 100:
        return 100
    return wanted


# -------------------- The page --------------------
PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Similarity</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        defer></script>
<style>
  :root {
    --brand: #e8590c;
    --brand-dark: #d9480f;
    --brand-tint: #fff4e6;
    /* Bootstrap paints links with the *rgb triplet* variables, not with
       --bs-link-color, so both forms have to be redirected or <a> stays
       #0d6efd blue. */
    --bs-link-color: #d9480f;
    --bs-link-hover-color: #c2410c;
    --bs-link-color-rgb: 217, 72, 15;
    --bs-link-hover-color-rgb: 194, 65, 12;
    /* Everything else in Bootstrap that ships as blue (#0d6efd / #cfe2ff). */
    --bs-primary-rgb: 232, 89, 12;
    --bs-focus-ring-color: rgba(232, 89, 12, .25);
    --bs-table-accent-bg: #fff4e6;
    --bs-table-active-bg: #fff4e6;
    --bs-secondary-bg: #f6f3f0;
    --bs-tertiary-bg: #f6f3f0;
    --bs-secondary-color: #9a8577;
  }
  /* Belt and braces: state the link colours directly so no cascade order or
     later Bootstrap rule can leave an anchor blue. */
  a, a:link, a:visited, a:hover, a:focus, a:active {
    color: #d9480f;
  }
  a:hover, a:focus { color: #c2410c; }
  /* The browser paints ::selection blue by default. */
  ::selection { background: #f3d6c0; color: #7a2e0a; }
  /* Every focusable element, in case Bootstrap's ring is overridden upstream. */
  a:focus-visible, .btn:focus-visible, .btn-close:focus-visible,
  .form-control:focus-visible, .form-select:focus-visible,
  input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline: none;
    box-shadow: 0 0 0 .25rem rgba(232, 89, 12, .25);
  }
  /* The file picker button is a real control on this page; it is blue by
     default in Bootstrap. */
  input[type="file"]::file-selector-button {
    background: #fff4e6; color: #7a2e0a;
    border: 1px solid #f3d6c0; border-radius: 6px;
    padding: .35rem .8rem; margin-right: .75rem;
  }
  input[type="file"]::file-selector-button:hover {
    background: #f3d6c0; border-color: #e8590c; color: #7a2e0a;
  }
  /* Unused on this page, overridden anyway so no future markup can leak blue. */
  .form-check-input:checked {
    background-color: #e8590c; border-color: #e8590c;
  }
  .form-check-input:focus {
    border-color: #e8590c; box-shadow: 0 0 0 .25rem rgba(232, 89, 12, .25);
  }
  .alert-link, .page-link { color: #c2410c; }
  html, body { height: 100%; }
  body {
    background: #f6f3f0;
    font-family: system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  /* Keep every accent orange-red: no blue anywhere. */
  .btn { --bs-btn-focus-shadow-rgb: 232, 89, 12; }
  .btn-primary {
    --bs-btn-color: #fff;
    --bs-btn-bg: #e8590c;
    --bs-btn-border-color: #e8590c;
    --bs-btn-hover-color: #fff;
    --bs-btn-hover-bg: #d9480f;
    --bs-btn-hover-border-color: #d9480f;
    --bs-btn-active-color: #fff;
    --bs-btn-active-bg: #c2410c;
    --bs-btn-active-border-color: #c2410c;
    --bs-btn-disabled-color: #fff;
    --bs-btn-disabled-bg: #f3a06a;
    --bs-btn-disabled-border-color: #f3a06a;
    --bs-btn-focus-shadow-rgb: 232, 89, 12;
  }
  .panel {
    background: #fff;
    border: 1px solid #ece4dc;
    border-top: 4px solid var(--brand);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(60, 30, 10, .06);
    height: 100%;
  }
  .panel-body { padding: 1.5rem 1.5rem 1.75rem; }
  .panel-title {
    font-size: 1.02rem; font-weight: 700; color: #7a2e0a;
    display: flex; align-items: center; gap: .55rem;
  }
  .step {
    width: 1.75rem; height: 1.75rem; border-radius: 50%;
    background: var(--brand); color: #fff;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: .9rem; font-weight: 700; flex: none;
  }
  .form-control:focus, .form-select:focus {
    border-color: var(--brand);
    box-shadow: 0 0 0 .25rem rgba(232, 89, 12, .22);
  }
  .form-text { color: #9a8577; }
  .btn-lg { padding: .7rem 1.4rem; font-size: 1rem; font-weight: 600; }
  .table thead th {
    background: var(--brand-tint); color: #7a2e0a;
    border-bottom: 2px solid #f3d6c0; font-size: .82rem;
    text-transform: uppercase; letter-spacing: .04em;
  }
  .table td { vertical-align: middle; }
  .badge-score {
    background: var(--brand); color: #fff;
    font-size: .95rem; font-weight: 700; min-width: 5.2rem;
  }
  .bar { height: 6px; border-radius: 3px; background: #f1e2d6; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: var(--brand); }
  .alert-brand { background: var(--brand-tint); border: 1px solid #f3c9a8; color: #7a2e0a; }
  .empty-state { color: #a8968a; font-size: .92rem; }
  .thumb { max-height: 88px; border-radius: 6px; border: 1px solid #e6dbd1; }
  .spinner {
    width: 1rem; height: 1rem; border: 2px solid rgba(255,255,255,.45);
    border-top-color: #fff; border-radius: 50%;
    display: inline-block; animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .verdict {
    background: var(--brand-tint); border: 1px solid #f3c9a8;
    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
  }
  .verdict .who { font-size: 1.3rem; font-weight: 700; color: #7a2e0a; }
</style>
</head>
<body>
<div class="container-fluid px-2 py-3">

  <div class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-3 px-2">
    <div>
      <h1 class="h3 mb-1" style="color:#7a2e0a">Face Similarity</h1>
      <div class="text-secondary">Upload Comparison &middot; Library Search</div>
    </div>
    <div class="text-end">
      <span class="badge rounded-pill text-bg-light border" id="libraryBadge">Loading library&hellip;</span>
    </div>
  </div>

  <div class="row g-3 align-items-stretch">
    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">1</span>Query image vs candidate images (1:N)</div>
          <p class="form-text mb-3">One query image, compared one by one with the candidate images picked below. The local face library is <strong>not</strong> used.</p>
          <form id="queryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="queryFile">① Query image (single)</label>
              <input class="form-control form-control-lg" type="file" id="queryFile"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="queryFile">e.g. who-is-this.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="queryFile"></div>
            </div>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="candidateFiles">② Candidate images (multiple)</label>
              <input class="form-control form-control-lg" type="file" id="candidateFiles"
                     name="candidates" accept="image/*" multiple required>
              <div class="form-text" data-summary="candidateFiles">e.g. user1.jpg, user2.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="candidateFiles"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="queryTop">Results to return</label>
              <input class="form-control form-control-lg" type="number" id="queryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="querySubmit">
              Start comparison
            </button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">2</span>Query image vs face library (1:N)</div>
          <p class="form-text mb-3">One query image, compared with <b>every</b> image in the face library; the closest matches are returned automatically.</p>
          <form id="libraryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="libraryQuery">① Query image (single)</label>
              <input class="form-control form-control-lg" type="file" id="libraryQuery"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="libraryQuery">e.g. who-is-this.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="libraryQuery"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="libraryTop">Results to return</label>
              <input class="form-control form-control-lg" type="number" id="libraryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="librarySubmit">
              Search library
            </button>
          </form>
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3 mt-0">
    <div class="col-12">
      <div class="panel">
        <div class="panel-body">
          <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
            <div class="panel-title mb-0"><span class="step">3</span>Results</div>
            <div class="form-text" id="resultMeta"></div>
          </div>
          <div id="resultAlert"></div>
          <div id="resultBody" class="empty-state">No comparison has been run yet.</div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function setLoading(button, loading, idleText) {
  button.disabled = loading;
  button.innerHTML = loading
    ? '<span class="spinner me-2"></span>Comparing&hellip;'
    : idleText;
}

function showError(message) {
  $("resultAlert").innerHTML =
    '<div class="alert alert-brand alert-dismissible fade show" role="alert">' +
    '<strong>Error: </strong>' + escapeHtml(message) +
    '<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>';
  $("resultBody").innerHTML = "";
  $("resultMeta").textContent = "";
}

function resultTable(rows) {
  if (!rows || !rows.length) {
    return '<p class="empty-state mb-0">Nothing to compare.</p>';
  }
  // The bar length is the similarity itself on a fixed 0-100 scale, so it
  // always matches the percentage printed next to it.
  const body = rows.map((row) => {
    const span = Math.max(0, Math.min(100, row.similarity));
    return `
    <tr${row.best ? ' class="table-warning"' : ''}>
      <td style="width:4rem">${row.rank}</td>
      <td>${escapeHtml(row.candidate)}</td>
      <td style="width:9rem">
        <span class="badge badge-score">${row.similarity.toFixed(2)}%</span>
      </td>
      <td style="width:32%">
        <div class="bar"><span style="width:${span}%"></span></div>
      </td>
    </tr>`;
  }).join("");
  return `<table class="table table-sm align-middle mb-0">
    <thead><tr><th>Rank</th><th>${escapeHtml(rows[0].kind || "Candidate")}</th><th>Similarity</th><th></th></tr></thead>
    <tbody>${body}</tbody></table>
    <p class="form-text mt-2 mb-0">The bar length is the similarity percentage; a full bar is 100%.</p>`;
}

function skippedList(skipped) {
  if (!skipped || !skipped.length) return "";
  return '<p class="form-text mt-3 mb-0">Skipped: '
    + skipped.map((item) => escapeHtml(item.name) + " (" + escapeHtml(item.status) + ")").join(", ")
    + '</p>';
}

function verdictBlock(matches) {
  if (!matches || !matches.length) {
    return '<p class="empty-state">No usable candidates.</p>';
  }
  const best = matches[0];
  return `<div class="verdict d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div>
        <div class="form-text mb-1">Closest match</div>
        <div class="who">${escapeHtml(best.candidate)}</div>
      </div>
      <div class="text-end">
        <div class="form-text mb-1">Similarity</div>
        <div style="font-size:2.4rem;color:#7a2e0a">${best.similarity.toFixed(2)}%</div>
      </div>
    </div>`;
}

function renderSingle(payload) {
  $("resultBody").innerHTML =
    verdictBlock(payload.matches) + resultTable(payload.matches)
    + skippedList(payload.skipped);
}

function renderPayload(payload) {
  $("resultAlert").innerHTML = "";
  const meta = [];
  if (payload.query) meta.push("Query: " + payload.query);
  if (payload.elapsed_ms != null) meta.push("Total " + payload.elapsed_ms + " ms");
  if (payload.count != null) meta.push("Processed " + payload.count);
  if (payload.library_faces != null) meta.push("Library " + payload.library_faces + " faces");
  if (payload.library_images != null) meta.push("Library " + payload.library_images + " images");
  $("resultMeta").textContent = meta.join(" · ");

  renderSingle(payload);
}

async function postJson(url, formData, button, idleText) {
  setLoading(button, true, idleText);
  $("resultAlert").innerHTML = "";
  try {
    const response = await fetch(url, { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch (parseError) {
      throw new Error("The server returned a response that could not be parsed (HTTP " + response.status + ")");
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || ("Request failed (HTTP " + response.status + ")"));
    }
    renderPayload(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(button, false, idleText);
  }
}

function bindForm(formId, buttonId, url, idleText, requiredIds) {
  $(formId).addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    for (const id of requiredIds) {
      if (!form.querySelector("#" + id).files.length) {
        showError("Please choose the images this panel needs first");
        return;
      }
    }
    postJson(url, new FormData(form), $(buttonId), idleText);
  });
}

bindForm("queryForm", "querySubmit", "/api/query-set", "Start comparison",
         ["queryFile", "candidateFiles"]);
bindForm("libraryForm", "librarySubmit", "/api/query-library", "Search library",
         ["libraryQuery"]);

document.querySelectorAll('input[type="file"]').forEach((input) => {
  input.addEventListener("change", () => {
    const picked = Array.from(input.files);
    const summary = document.querySelector('[data-summary="' + input.id + '"]');
    if (summary && input.multiple) {
      summary.textContent = picked.length
        ? picked.length + " images selected"
        : "Nothing selected yet";
    } else if (summary && picked.length) {
      summary.textContent = "Selected: " + picked[0].name;
    }
    const preview = document.querySelector('[data-preview="' + input.id + '"]');
    if (!preview) return;
    const tiles = picked.slice(0, 8).map((file) =>
      '<img class="thumb" src="' + URL.createObjectURL(file) + '" title="'
      + escapeHtml(file.name) + '">');
    if (picked.length > 8) {
      tiles.push('<span class="thumb d-flex align-items-center px-2 text-secondary">+'
        + (picked.length - 8) + '</span>');
    }
    preview.innerHTML = tiles.join("");
  });
});

function loadLibraryStatus() {
  return fetch("/api/library")
    .then((response) => response.json())
    .then((payload) => {
      $("libraryBadge").textContent = payload.ok
        ? "Library: " + payload.faces + " faces / " + payload.images + " images"
        : "Library unavailable";
    })
    .catch(() => { $("libraryBadge").textContent = "Library status unknown"; });
}

loadLibraryStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """The page itself, both panels included. There is no detection mode
    picker on this page.

    The three numbers are the initial value and the bounds of the "results to
    return" input, kept in step with the limits in top_results_from_request.
    """
    return render_template_string(PAGE, top_results=10, top_results_min=1, top_results_max=100)


@app.get("/api/library")
def api_library():
    """How many images and faces the library holds.

    Only the counts are returned; the folder path itself never leaves the
    server.
    """
    _labels, embedding_matrix = load_library()
    return jsonify(
        {
            "ok": True,
            "images": len(library_images()),
            "faces": int(embedding_matrix.shape[0]),
        }
    )


@app.post("/api/query-set")
def api_query_set():
    """Panel 1: the query image against the candidate images uploaded with it."""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    uploads = read_uploads("candidates")
    if not uploads:
        raise RequestError("Please choose at least one candidate image")

    # An unusable image is recorded and skipped; the whole comparison goes on.
    labels = []
    candidate_faces = []
    skipped = []
    for name, data in uploads:
        try:
            faces = encode_upload(data, "Candidate image", name)
        except RequestError as error:
            skipped.append({"name": name, "status": str(error)})
            continue
        labels.append(name)
        candidate_faces.append(faces[0])        # the biggest face in that photo

    if not candidate_faces:
        raise RequestError("No usable face in the candidate images")

    return jsonify(
        {
            "ok": True,
            "mode": "query-set",
            "query": query_name,
            "query_faces": len(query_faces),
            "count": len(uploads),
            "compared": len(candidate_faces),
            "matches": rank_matches(
                query_faces,
                disambiguate(labels),
                candidate_faces,
                "Uploaded candidate",
                top_results_from_request(),
            ),
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.post("/api/query-library")
def api_query_library():
    """Panel 2: the query image against every image of the face library."""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    if not library_images():
        raise RequestError("The face library has no images; add some images first and try again")

    labels, embedding_matrix = load_library()
    if embedding_matrix.shape[0] == 0:
        raise RequestError("No face detected in the face library; add images that contain faces first and try again")

    return jsonify(
        {
            "ok": True,
            "mode": "query-library",
            "query": query_name,
            "query_faces": len(query_faces),
            "library_images": len(library_images()),
            "library_faces": int(embedding_matrix.shape[0]),
            "matches": rank_matches(
                query_faces,
                labels,
                embedding_matrix,
                "Library candidate",
                top_results_from_request(),
            ),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


# =============================================================================
# Startup
# =============================================================================


def ask_library_folder() -> Path:
    """Ask once at startup where the face library is; pressing Enter uses
    face_library next to this script.

    When there is no window to ask in (double clicked, or started by another
    program) nothing is asked and the default location is used, rather than
    reading from an input stream that is already closed.
    """
    default = SCRIPT_DIRECTORY / "face_library"
    if not sys.stdin.isatty():
        default.mkdir(parents=True, exist_ok=True)
        return default

    print("The face library is just a folder of images; subfolders are fine.")
    print(f"Supported formats: {' '.join(sorted(SUPPORTED_EXTENSIONS))}")
    while True:
        answer = input(f"Face library folder (press Enter for {default}): ").strip()
        if not answer:
            default.mkdir(parents=True, exist_ok=True)
            return default
        folder = Path(answer).expanduser()
        if folder.is_dir():
            return folder
        if folder.exists():
            print("That is a file, not a folder. Please enter it again.")
        else:
            print("That folder does not exist. Please check the path and enter it again.")


def choose_port(preferred: int = 5000) -> int:
    """Find a free port at or after the preferred one.

    Port 5000 is often taken by something else, so the next one is used
    automatically instead of failing to start.
    """
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # No SO_REUSEADDR here: on Windows it would let this probe bind a
            # port another process is already listening on.
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port between {preferred} and {preferred + 19}")


def main():
    """Ask where the face library is, load the models, then serve the page."""
    global LIBRARY_FOLDER
    LIBRARY_FOLDER = ask_library_folder()

    print("Loading the buffalo_l models, about 2 seconds...", flush=True)
    load_model()
    print("Models loaded")
    print(MODEL_SUMMARY, flush=True)
    print(f"Face library: {LIBRARY_FOLDER}", flush=True)

    images = library_images()
    print(f"The face library currently holds {len(images)} images", flush=True)

    port = choose_port()
    print(f"Open http://127.0.0.1:{port} in your browser", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()