"""
Flask web UI for face similarity: two panels side by side, everything AJAX.

    1. 查询图 vs 候选图片    one query image against the images picked in the same
                          request (1:N). The face library is not touched.
    2. 查询图 vs 人脸库      one query image against every image of the face
                          library (1:N).

Scoring
    Every face becomes a 128-d descriptor (resnet_model_v1, L2 normalized) and
    the score is the cosine similarity in percent. On that scale two photos of
    the same person usually land above 85 and two different people lower. This
    file never decides that two faces are the same person: pick that threshold
    yourself, from your own samples.

Detection
    The visitor picks 准确 (CNN) or 快速 (HOG). That detector is the only
    difference between the two modes: the landmarks model, the descriptor and
    the score are the same code either way, so the numbers stay comparable.
    Each mode caches its own descriptors, so switching back and forth never
    re-encodes the library.

Configuration
    Command line only, never environment variables, so a noisy environment
    cannot change how this runs:
        --dir D:\\faces   the face library folder
                         (default: face_library/ next to this file)
        --port 5000      first port to try; a busy one is skipped
        --hog           make 快速 the default mode on the page
        --library-info   count the library, print it, exit without serving
    The page neither asks for nor shows that folder, so where the faces live is
    not something a visitor can see or change.

Self contained
    Detection, descriptor and scoring are embedded in this file: no second
    module to import, no second file to upload next to this one. The .dat model
    files are separate data and are found automatically, either next to this
    file (models/, this folder, or any subfolder of it) or in an installed
    face_recognition_models package.

Run:
    python flask_face_match_v2.py
    python flask_face_match_v2.py --port 5000 --dir D:\\faces
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import dlib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

# =============================================================================
# Configuration
#
# Constants of the program, not knobs of the machine: changing one changes how
# the program behaves, on every computer. The two values the command line may
# override live in Settings at the bottom of this section.
# =============================================================================

# --- model files ---
# The landmarks and the recognition network produce the score, so they are
# always needed. The CNN file is only opened in 准确 mode.
LANDMARKS_FILE = "shape_predictor_68_face_landmarks.dat"
LANDMARKS_FALLBACK_FILE = "shape_predictor_5_face_landmarks.dat"
RECOGNITION_FILE = "dlib_face_recognition_resnet_model_v1.dat"
CNN_DETECTOR_FILE = "mmod_human_face_detector.dat"

DESCRIPTOR_DIMENSION = 128

# --- detection modes ---
DETECTOR_CNN = "cnn"                  # 准确: finds angled and small faces, slow
DETECTOR_HOG = "hog"                  # 快速: much cheaper, needs frontal faces
DETECTOR_NAMES = (DETECTOR_CNN, DETECTOR_HOG)

# How many times the image is doubled before detection. This is the only
# detection sensitivity knob the Python binding offers: 0 keeps the original
# size, and every step multiplies the time by about four while also finding
# smaller faces. HOG needs that step to be usable, CNN is accurate without it.
DETECTOR_UPSAMPLE = {DETECTOR_CNN: 0, DETECTOR_HOG: 1}

# --- face library ---
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
DEFAULT_LIBRARY_FOLDER_NAME = "face_library"
LIBRARY_PROGRESS_FROM = 20            # print per image progress from this count

# --- web service ---
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_TOP_RESULTS = 10
MIN_TOP_RESULTS = 1
MAX_TOP_RESULTS = 100
DEFAULT_PORT = 5000
PORT_SEARCH_ATTEMPTS = 20

# --- form field names, shared by the page and the endpoints ---
QUERY_FIELD = "query"
CANDIDATE_FIELD = "candidates"
DETECTOR_FIELD = "detector"
TOP_RESULTS_FIELD = "top_results"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


@dataclass
class Settings:
    """What the command line asked for. Filled in once, then read-only."""

    library_directory: str = ""
    default_detector: str = DETECTOR_CNN


SETTINGS = Settings()


def default_library_directory() -> Path:
    """The library folder this server searches.

    --dir sets it; the page never names a folder, so this is the only place
    that decides, and the path never leaves the server.
    """
    if SETTINGS.library_directory:
        return Path(SETTINGS.library_directory).expanduser()
    return SCRIPT_DIRECTORY / DEFAULT_LIBRARY_FOLDER_NAME


def default_detector() -> str:
    """The mode the page offers first; --hog changes it to the fast one."""
    return normalize_detector(SETTINGS.default_detector)


# =============================================================================
# Finding the model files
#
# dlib opens models by path, and on Windows it refuses paths that contain non
# ASCII characters, so a file is mirrored into the temp folder when needed.
# =============================================================================


def _is_ascii_safe(path: Path) -> bool:
    """True when dlib will be able to open this path."""
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _model_search_paths(file_name: str) -> list:
    """Every place the model file may live, most specific location first.

    Next to this file wins (models/, this folder, any folder below it), then
    the models folder of a face_recognition_models package on sys.path.
    """
    candidates = [
        SCRIPT_DIRECTORY / "models" / file_name,
        SCRIPT_DIRECTORY / file_name,
        *sorted(SCRIPT_DIRECTORY.rglob(file_name)),
    ]
    for entry in sys.path:
        base_directory = Path(entry) if entry else Path.cwd()
        candidates.append(base_directory / file_name)
        candidates.append(
            base_directory / "face_recognition_models" / "models" / file_name
        )

    existing: list = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in existing:
            existing.append(candidate)
    return existing


def find_model_file(file_name: str) -> Path:
    """Locate one model file without importing any helper package."""
    candidates = _model_search_paths(file_name)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {file_name}. Put it in a models folder next to "
            f"this file, next to this file itself, or in the models folder of "
            f"an installed face_recognition_models package."
        )
    for candidate in candidates:
        if _is_ascii_safe(candidate):
            return candidate
    return candidates[0]


@lru_cache(maxsize=8)
def dlib_loadable_path(path: str) -> str:
    """Return a path dlib can actually open, mirroring non-ASCII ones."""
    model_path = Path(path)
    if _is_ascii_safe(model_path):
        return str(model_path)

    cache_directory = Path(tempfile.gettempdir()) / "dlib_face_models"
    cache_directory.mkdir(parents=True, exist_ok=True)
    mirrored_path = cache_directory / model_path.name
    if (
        not mirrored_path.is_file()
        or mirrored_path.stat().st_size != model_path.stat().st_size
    ):
        shutil.copyfile(model_path, mirrored_path)
    return str(mirrored_path)


def _open_model(loader, path: Path):
    """Build one dlib model object from a file, mirroring it when needed."""
    return loader(dlib_loadable_path(str(path)))


def _landmarks_path() -> tuple:
    """The 68 point landmarks file, or the 5 point one when only it is there.

    Both predict a face for the recognition network, so the smaller one is an
    acceptable stand in rather than a failure.
    """
    try:
        return LANDMARKS_FILE, find_model_file(LANDMARKS_FILE)
    except FileNotFoundError:
        return LANDMARKS_FALLBACK_FILE, find_model_file(LANDMARKS_FALLBACK_FILE)


# =============================================================================
# Face models
#
# One detection mode needs three networks: a detector, the 68 point landmark
# predictor and the recognition network. Both modes share the last two, which
# is why the scores of 准确 and 快速 can be compared with each other.
#
# Loading costs about a second, so each mode is loaded once and then kept for
# the life of the process; a visitor may switch modes without a restart.
# =============================================================================


def normalize_detector(name: str = "") -> str:
    """Anything that is not "hog" means the CNN detector."""
    return DETECTOR_HOG if str(name or "").strip().lower() == DETECTOR_HOG else DETECTOR_CNN


@dataclass(frozen=True)
class FaceModels:
    """The three networks one detection mode needs, plus the two operations
    that use them: finding faces, and describing one face."""

    mode: str
    detector: object
    shape_predictor: object
    recognizer: object
    landmarks_file: str

    def detect(self, rgb_array: np.ndarray) -> list:
        """Every face in an RGB image, as plain rectangles, largest first.

        The CNN detector reports a confidence and wraps its boxes in
        mmod_rectangle objects, so those wrappers are dropped here; the HOG
        detector already returns plain boxes.
        """
        upsample = DETECTOR_UPSAMPLE[self.mode]
        if self.mode == DETECTOR_CNN:
            found = [item.rect for item in self.detector(rgb_array, upsample)]
        else:
            found = list(self.detector(rgb_array, upsample))

        # Biggest face first, so the caller can treat element 0 as *the* face
        # of a photo without asking how many there were.
        found.sort(key=lambda box: box.width() * box.height(), reverse=True)
        return found

    def describe(self, rgb_array: np.ndarray, rectangle) -> np.ndarray:
        """One L2 normalized 128-d descriptor for one detected face box."""
        height, width = rgb_array.shape[:2]
        landmarks = self.shape_predictor(
            rgb_array, _clamp_rectangle(rectangle, height, width)
        )
        descriptor = np.array(
            self.recognizer.compute_face_descriptor(rgb_array, landmarks),
            dtype="float32",
        )
        norm = float(np.linalg.norm(descriptor))
        if norm == 0.0:
            raise ValueError("人脸描述子为空")
        return descriptor / norm


_MODEL_CACHE: dict[str, FaceModels] = {}
_MODEL_LOCK = threading.Lock()


def _load_models(mode: str) -> FaceModels:
    """Read the model files of one detection mode from disk."""
    landmarks_file, landmarks_path = _landmarks_path()

    if mode == DETECTOR_CNN:
        face_detector = _open_model(
            dlib.cnn_face_detection_model_v1, find_model_file(CNN_DETECTOR_FILE)
        )
    else:
        # The HOG detector is built into dlib, it has no model file.
        face_detector = dlib.get_frontal_face_detector()

    return FaceModels(
        mode=mode,
        detector=face_detector,
        shape_predictor=_open_model(dlib.shape_predictor, landmarks_path),
        recognizer=_open_model(
            dlib.face_recognition_model_v1, find_model_file(RECOGNITION_FILE)
        ),
        landmarks_file=landmarks_file,
    )


def get_models(mode: str = DETECTOR_CNN) -> FaceModels:
    """The models of one detection mode, loaded on first use and kept."""
    mode = normalize_detector(mode)
    with _MODEL_LOCK:
        if mode not in _MODEL_CACHE:
            _MODEL_CACHE[mode] = _load_models(mode)
        return _MODEL_CACHE[mode]


def warm_up(mode: str = DETECTOR_CNN) -> float:
    """Load the models and report how many milliseconds that took."""
    started = time.perf_counter()
    get_models(mode)
    return (time.perf_counter() - started) * 1000.0


def describe_configuration() -> str:
    """One line summary of modes and score scale, for the server log."""
    loaded = ", ".join(
        f"{mode}={_MODEL_CACHE[mode].landmarks_file}"
        for mode in DETECTOR_NAMES
        if mode in _MODEL_CACHE
    ) or "not loaded yet"
    return (
        f"detector={default_detector()} dim={DESCRIPTOR_DIMENSION} "
        f"landmarks=[{loaded}] recognition={RECOGNITION_FILE} "
        f"library={default_library_directory()}"
    )


# =============================================================================
# Turning input into pixels
#
# Everything downstream of this section works on one shape only: a contiguous
# RGB uint8 NumPy array, which is what dlib wants.
# =============================================================================

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def _image_bytes(image: ImageInput) -> bytes:
    """The encoded bytes of any supported input."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)

    image_path = Path(image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    # The bytes are read here because dlib cannot open non-ASCII paths on
    # Windows; decoding happens below, in one place, for every input kind.
    return image_path.read_bytes()


def to_rgb_array(image: ImageInput, array_is_bgr: bool = True) -> np.ndarray:
    """Normalize any supported input into a contiguous RGB uint8 array.

    Args:
        image: file path, encoded image bytes, or a NumPy array.
        array_is_bgr: set to False when the NumPy array is already RGB.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            array = np.repeat(image[:, :, None], 3, axis=2)       # gray to RGB
        else:
            array = image[:, :, ::-1] if array_is_bgr else image  # BGR to RGB
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype("uint8")
        return np.ascontiguousarray(array)

    # opencv decodes to BGR whatever the file format was.
    decoded = cv2.imdecode(
        np.frombuffer(_image_bytes(image), dtype="uint8"), cv2.IMREAD_COLOR
    )
    if decoded is None:
        raise ValueError("Unsupported or corrupted image data")
    return np.ascontiguousarray(decoded[:, :, ::-1])


def _clamp_rectangle(rectangle, rows: int, columns: int):
    """Keep a detection box inside the image bounds.

    A box can stick out by a pixel on photos that were resized or cropped;
    dlib raises on such a box instead of clipping it.
    """
    left = max(0, min(int(rectangle.left()), columns - 1))
    top = max(0, min(int(rectangle.top()), rows - 1))
    right = max(left + 1, min(int(rectangle.right()), columns))
    bottom = max(top + 1, min(int(rectangle.bottom()), rows))
    return dlib.rectangle(left, top, right, bottom)


# =============================================================================
# Encoding and scoring
#
# These four functions are the public surface of the engine; the web service
# above them only moves images and JSON around.
# =============================================================================


def encode_faces(
    image: ImageInput, array_is_bgr: bool = True, detector: str = DETECTOR_CNN
) -> list:
    """Return one L2 normalized 128-d descriptor per detected face.

    Sorted by detection box area from large to small, so element 0 is the
    biggest face in the image. Returns an empty list when no face is found.
    """
    models = get_models(detector)
    rgb_array = to_rgb_array(image, array_is_bgr)
    return [models.describe(rgb_array, box) for box in models.detect(rgb_array)]


def cosine_similarity_percent(
    descriptor_a: np.ndarray, descriptor_b: np.ndarray
) -> float:
    """Cosine similarity of two descriptors, expressed as a percentage."""
    first = np.asarray(descriptor_a, dtype="float32").reshape(-1)
    second = np.asarray(descriptor_b, dtype="float32").reshape(-1)
    if first.size != second.size:
        raise ValueError(f"Descriptor sizes differ: {first.size} and {second.size}")
    return float(np.dot(first, second)) * 100.0


def similarity_matrix_percent(
    query_descriptors: Sequence, candidate_descriptors: Sequence
) -> np.ndarray:
    """Best score for every candidate, as a 1-D array of percentages.

    A query may hold several faces (a group photo), so each candidate is scored
    against the query face that matches it best.
    """
    query_matrix = np.vstack(query_descriptors).astype("float32")
    candidate_matrix = np.vstack(candidate_descriptors).astype("float32")
    return (candidate_matrix @ query_matrix.T).max(axis=1) * 100.0


def face_similarity_percent(
    image_a: ImageInput,
    image_b: ImageInput,
    array_is_bgr: bool = True,
    compare_all_faces: bool = False,
    detector: str = DETECTOR_CNN,
) -> Optional[float]:
    """Similarity between the faces in two images, as a percentage.

    Args:
        image_a: image path, encoded image bytes, or a NumPy array.
        image_b: image path, encoded image bytes, or a NumPy array.
        array_is_bgr: set to False when a NumPy array is already in RGB order.
        compare_all_faces: use the best matching face pair instead of the
            largest face of each image.
        detector: DETECTOR_CNN for accuracy, DETECTOR_HOG for speed.

    Returns:
        A float percentage, or None when no face is found in one of the images.
        Higher means more similar.
    """
    descriptors_a = encode_faces(image_a, array_is_bgr, detector)
    descriptors_b = encode_faces(image_b, array_is_bgr, detector)

    if not descriptors_a or not descriptors_b:
        return None

    if descriptors_a[0].size != DESCRIPTOR_DIMENSION:
        raise ValueError(f"Unexpected descriptor size: {descriptors_a[0].size}")

    if compare_all_faces:
        similarity = max(
            cosine_similarity_percent(first, second)
            for first in descriptors_a
            for second in descriptors_b
        )
    else:
        similarity = cosine_similarity_percent(descriptors_a[0], descriptors_b[0])

    # Guard against floating point drift outside the valid range.
    return float(min(100.0, max(-100.0, similarity)))


# =============================================================================
# The face library
#
# Any folder of images. Encoding a whole library with the CNN detector takes a
# while, so every file is encoded once and its descriptors are cached, keyed by
# file and detection mode and invalidated by the file's modified time: editing,
# adding or deleting an image is picked up on the next search, and switching
# between the two modes never re-encodes what the other mode already has.
# =============================================================================

# (path, mode) -> (modified time, descriptors)
_DESCRIPTOR_CACHE: dict = {}
_LIBRARY_LOCK = threading.Lock()


def library_images(directory: Path) -> list:
    """Every image under a folder, recursively, in a stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _descriptors_of(path: Path, mode: str) -> list:
    """The descriptors of one library file, from cache when still valid.

    A file that cannot be read or holds no face yields an empty list rather
    than an error: one bad photo must not take the whole library down.
    """
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        return []          # deleted between the folder scan and now

    key = (str(path), mode)
    cached = _DESCRIPTOR_CACHE.get(key)
    if cached is not None and cached[0] == modified_time:
        return cached[1]

    try:
        descriptors = encode_faces(path, detector=mode)
    except (OSError, ValueError):
        descriptors = []
    _DESCRIPTOR_CACHE[key] = (modified_time, descriptors)
    return descriptors


def _forget_deleted_files(mode: str, alive_keys: set) -> None:
    """Drop the cached descriptors of files this mode no longer has.

    Only the given mode is touched: the other mode's cache stays valid and is
    cleaned when it is the one being searched.
    """
    for key in [key for key in _DESCRIPTOR_CACHE if key[1] == mode]:
        if key not in alive_keys:
            del _DESCRIPTOR_CACHE[key]


def load_library(directory: Path, detector: str = DETECTOR_CNN) -> tuple:
    """Return (labels, matrix) for every face in the library folder.

    Labels are file names, with " #2", " #3" appended when one photo holds
    several faces. The matrix is one 128-d row per label, in the same order.
    """
    mode = normalize_detector(detector)
    paths = library_images(directory)
    show_progress = len(paths) >= LIBRARY_PROGRESS_FROM

    labels: list = []
    rows: list = []
    alive_keys: set = set()

    # One search at a time: the CNN detector is single threaded inside dlib
    # and two encoders at once would only make both of them slower.
    with _LIBRARY_LOCK:
        for index, path in enumerate(paths, start=1):
            if show_progress:
                print(f"  载入人脸库 {index}/{len(paths)}  {path.name}", flush=True)

            alive_keys.add((str(path), mode))
            for position, descriptor in enumerate(_descriptors_of(path, mode)):
                labels.append(
                    path.name if position == 0 else f"{path.name} #{position + 1}"
                )
                rows.append(descriptor)

        _forget_deleted_files(mode, alive_keys)

    if not rows:
        return labels, np.zeros((0, DESCRIPTOR_DIMENSION), dtype="float32")
    return labels, np.vstack(rows).astype("float32")


# =============================================================================
# Turning scores into a result table
# =============================================================================


def rank_scores(
    labels: list, scores: np.ndarray, top_results: int, kind: str
) -> list:
    """Turn raw scores into a ranked, rounded list, best first.

    The bar drawn next to each row on the page is the similarity itself on a
    fixed 0-100 scale, so a bar is always as long as the number printed next
    to it.
    """
    order = np.argsort(-scores)[: max(1, int(top_results))]
    return [
        {
            "rank": rank,
            "kind": kind,
            "candidate": labels[int(index)],
            "similarity": round(float(scores[int(index)]), 2),
        }
        for rank, index in enumerate(order, start=1)
    ]


def disambiguate(names: list) -> list:
    """Number repeated names, so two rows never show the same label twice."""
    totals: dict = {}
    for name in names:
        totals[name] = totals.get(name, 0) + 1

    labels: list = []
    seen: dict = {}
    for name in names:
        if totals[name] == 1:
            labels.append(name)
            continue
        seen[name] = seen.get(name, 0) + 1
        labels.append(f"{name} ({seen[name]})")
    return labels


# =============================================================================
# Web service
#
# Three endpoints, all stateless: the page posts an image, the endpoint scores
# it and answers with JSON. Errors travel as RequestError, which one handler
# turns into the {"ok": false, "error": ...} body the page expects.
# =============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


class RequestError(Exception):
    """Something about this request is wrong; the message is shown to the user."""


class ImageRejected(Exception):
    """This image cannot be compared; the message is shown to the user."""


def failure(message: str, status: int = 400):
    """The one error body both the page and a person reading the API see."""
    return jsonify({"ok": False, "error": message}), status


@app.errorhandler(RequestError)
def handle_request_error(error: RequestError):
    return failure(str(error))


@app.errorhandler(413)
def handle_too_large(_error):
    return failure(f"上传文件太大，单次请求上限 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)


# -------------------- Reading a request --------------------
def read_uploads(field_name: str) -> list:
    """Every uploaded file of a field, as (filename, bytes) pairs."""
    return [
        (item.filename, item.read())
        for item in request.files.getlist(field_name)
        if item and item.filename
    ]


def detector_from_request() -> str:
    """Which mode this request asked for: DETECTOR_CNN or DETECTOR_HOG."""
    return normalize_detector(
        request.form.get(DETECTOR_FIELD) or request.args.get(DETECTOR_FIELD) or ""
    )


def top_results_from_request() -> int:
    """How many rows to return, clamped to what the page allows."""
    raw_value = request.form.get(TOP_RESULTS_FIELD) or request.args.get(
        TOP_RESULTS_FIELD
    )
    try:
        return max(MIN_TOP_RESULTS, min(MAX_TOP_RESULTS, int(raw_value)))
    except (TypeError, ValueError):
        return DEFAULT_TOP_RESULTS


def encode_upload(data: bytes, detector: str) -> list:
    """The descriptors of one uploaded image. Raises ImageRejected."""
    try:
        descriptors = encode_faces(data, detector=detector)
    except (OSError, ValueError) as error:
        raise ImageRejected(f"处理失败：{error}") from error
    if not descriptors:
        raise ImageRejected("未检测到人脸")
    return descriptors


def query_from_request(detector: str) -> tuple:
    """The single query image of this request, encoded. Raises RequestError."""
    uploads = read_uploads(QUERY_FIELD)
    if not uploads:
        raise RequestError("请选择一张查询图片")
    if len(uploads) > 1:
        raise RequestError("查询图片只能选一张")

    name, data = uploads[0]
    try:
        return name, encode_upload(data, detector)
    except ImageRejected as error:
        raise RequestError(f"查询图片{error}：{name}") from error


def rank_matches(
    query_descriptors: list, labels: list, candidates, kind: str, top_results: int
) -> list:
    """Score every candidate against the query faces and rank the best ones.

    The query may hold several faces (a group photo); a candidate is then
    scored against the query face that matches it best, which is what
    similarity_matrix_percent returns.
    """
    matches = rank_scores(
        labels,
        similarity_matrix_percent(query_descriptors, candidates),
        top_results,
        kind,
    )
    if matches:
        matches[0]["best"] = True
    return matches


# -------------------- The page --------------------
PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人脸相似度</title>
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
    font-family: system-ui, "Microsoft YaHei", "Segoe UI", sans-serif;
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
  .mode-picker { min-width: 9.5rem; }
</style>
</head>
<body>
<div class="container-fluid px-2 py-3">

  <div class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-3 px-2">
    <div>
      <h1 class="h3 mb-1" style="color:#7a2e0a">人脸相似度</h1>
      <div class="text-secondary">上传比对 · 人脸库检索</div>
    </div>
    <div class="text-end d-flex flex-wrap align-items-center justify-content-end gap-2">
      <label class="form-text mb-0" for="detectorMode">检测方式</label>
      <select class="form-select form-select-sm mode-picker" id="detectorMode">
        <option value="{{ detector_cnn }}"{{ ' selected' if detector != detector_hog else '' }}>准确（CNN）</option>
        <option value="{{ detector_hog }}"{{ ' selected' if detector == detector_hog else '' }}>快速</option>
      </select>
      <span class="badge rounded-pill text-bg-light border" id="libraryBadge">人脸库载入中…</span>
    </div>
  </div>

  <div class="row g-3 align-items-stretch">
    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">1</span>查询图 vs 候选图片（1:N）</div>
          <p class="form-text mb-3">一张查询图，逐个和下面选中的候选图比对，<strong>不使用</strong>本地人脸库。</p>
          <form id="queryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="queryFile">① 查询图片（单张）</label>
              <input class="form-control form-control-lg" type="file" id="queryFile"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="queryFile">例如：查询这是谁.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="queryFile"></div>
            </div>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="candidateFiles">② 候选图片（可多选）</label>
              <input class="form-control form-control-lg" type="file" id="candidateFiles"
                     name="candidates" accept="image/*" multiple required>
              <div class="form-text" data-summary="candidateFiles">例如：1用户1.jpg、2用户2.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="candidateFiles"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="queryTop">返回条数</label>
              <input class="form-control form-control-lg" type="number" id="queryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="querySubmit">
              开始比对
            </button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">2</span>查询图 vs 人脸库（1:N）</div>
          <p class="form-text mb-3">一张查询图，对比人脸库里的<b>全部</b>图片，自动返回最像的几张。</p>
          <form id="libraryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="libraryQuery">① 查询图片（单张）</label>
              <input class="form-control form-control-lg" type="file" id="libraryQuery"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="libraryQuery">例如：查询这是谁.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="libraryQuery"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="libraryTop">返回条数</label>
              <input class="form-control form-control-lg" type="number" id="libraryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="librarySubmit">
              搜索人脸库
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
            <div class="panel-title mb-0"><span class="step">3</span>结果</div>
            <div class="form-text" id="resultMeta"></div>
          </div>
          <div id="resultAlert"></div>
          <div id="resultBody" class="empty-state">尚未发起比对。</div>
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
    ? '<span class="spinner me-2"></span>比对中…'
    : idleText;
}

function showError(message) {
  $("resultAlert").innerHTML =
    '<div class="alert alert-brand alert-dismissible fade show" role="alert">' +
    '<strong>出错了：</strong>' + escapeHtml(message) +
    '<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>';
  $("resultBody").innerHTML = "";
  $("resultMeta").textContent = "";
}

function resultTable(rows) {
  if (!rows || !rows.length) {
    return '<p class="empty-state mb-0">没有可比较的对象。</p>';
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
    <thead><tr><th>排名</th><th>${escapeHtml(rows[0].kind || "候选")}</th><th>相似度</th><th></th></tr></thead>
    <tbody>${body}</tbody></table>
    <p class="form-text mt-2 mb-0">横条长度就是相似度百分比，满条为 100%。</p>`;
}

function skippedList(skipped) {
  if (!skipped || !skipped.length) return "";
  return '<p class="form-text mt-3 mb-0">跳过：'
    + skipped.map((item) => escapeHtml(item.name) + "（" + escapeHtml(item.status) + "）").join("、")
    + '</p>';
}

function verdictBlock(matches) {
  if (!matches || !matches.length) {
    return '<p class="empty-state">没有可用候选。</p>';
  }
  const best = matches[0];
  return `<div class="verdict d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div>
        <div class="form-text mb-1">最像的是</div>
        <div class="who">${escapeHtml(best.candidate)}</div>
      </div>
      <div class="text-end">
        <div class="form-text mb-1">相似度</div>
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
  if (payload.query) meta.push("查询：" + payload.query);
  if (payload.elapsed_ms != null) meta.push("总耗时 " + payload.elapsed_ms + " ms");
  if (payload.count != null) meta.push("已处理 " + payload.count + " 张");
  if (payload.library_faces != null) meta.push("人脸库 " + payload.library_faces + " 张脸");
  if (payload.library_images != null) meta.push("人脸库 " + payload.library_images + " 张图");
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
      throw new Error("服务器返回了无法解析的内容（HTTP " + response.status + "）");
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || ("请求失败（HTTP " + response.status + "）"));
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
        showError("请先选择这个面板需要的图片");
        return;
      }
    }
    const body = new FormData(form);
    // The mode picker sits outside both forms, so it is added by hand.
    body.append("detector", $("detectorMode").value);
    postJson(url, body, $(buttonId), idleText);
  });
}

bindForm("queryForm", "querySubmit", "/api/query-set", "开始比对",
         ["queryFile", "candidateFiles"]);
bindForm("libraryForm", "librarySubmit", "/api/query-library", "搜索人脸库",
         ["libraryQuery"]);

document.querySelectorAll('input[type="file"]').forEach((input) => {
  input.addEventListener("change", () => {
    const picked = Array.from(input.files);
    const summary = document.querySelector('[data-summary="' + input.id + '"]');
    if (summary && input.multiple) {
      summary.textContent = picked.length
        ? "已选择 " + picked.length + " 张图片"
        : "尚未选择";
    } else if (summary && picked.length) {
      summary.textContent = "已选择：" + picked[0].name;
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
  return fetch("/api/library?detector=" + encodeURIComponent($("detectorMode").value))
    .then((response) => response.json())
    .then((payload) => {
      $("libraryBadge").textContent = payload.ok
        ? "人脸库 " + payload.faces + " 张人脸 / " + payload.images + " 张图"
        : "人脸库不可用";
    })
    .catch(() => { $("libraryBadge").textContent = "人脸库状态未知"; });
}

loadLibraryStatus();

// Each mode has its own cached descriptors, so the counts are asked again
// whenever the visitor switches between fast and accurate.
$("detectorMode").addEventListener("change", loadLibraryStatus);
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """The page itself: both panels and the mode picker."""
    return render_template_string(
        PAGE,
        detector=default_detector(),
        detector_cnn=DETECTOR_CNN,
        detector_hog=DETECTOR_HOG,
        top_results=DEFAULT_TOP_RESULTS,
        top_results_min=MIN_TOP_RESULTS,
        top_results_max=MAX_TOP_RESULTS,
    )


@app.get("/api/library")
def api_library():
    """How many images and faces the library holds, in the asked for mode.

    Only the counts are returned; the folder itself never leaves the server.
    """
    directory = default_library_directory()
    _labels, matrix = load_library(directory, detector_from_request())
    return jsonify(
        {
            "ok": True,
            "images": len(library_images(directory)),
            "faces": int(matrix.shape[0]),
        }
    )


@app.post("/api/query-set")
def api_query_set():
    """Panel 1: the query image against every image uploaded in this request."""
    started = time.perf_counter()
    detector = detector_from_request()
    query_name, query_descriptors = query_from_request(detector)

    candidates = read_uploads(CANDIDATE_FIELD)
    if not candidates:
        raise RequestError("请至少选择一张候选图片")

    # One unusable photo is reported and skipped, it does not fail the search.
    labels: list = []
    rows: list = []
    skipped: list = []
    for name, data in candidates:
        try:
            descriptors = encode_upload(data, detector)
        except ImageRejected as error:
            skipped.append({"name": name, "status": str(error)})
            continue
        labels.append(name)
        rows.append(descriptors[0])          # biggest face of that photo

    if not rows:
        raise RequestError("候选图片里没有可用的人脸")

    return jsonify(
        {
            "ok": True,
            "mode": "query-set",
            "query": query_name,
            "query_faces": len(query_descriptors),
            "detector": detector,
            "count": len(candidates),
            "compared": len(rows),
            "matches": rank_matches(
                query_descriptors,
                disambiguate(labels),
                rows,
                "候选图片",
                top_results_from_request(),
            ),
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.post("/api/query-library")
def api_query_library():
    """Panel 2: the query image against every image of the face library."""
    started = time.perf_counter()
    detector = detector_from_request()
    query_name, query_descriptors = query_from_request(detector)

    directory = default_library_directory()
    images = library_images(directory)
    if not images:
        raise RequestError("人脸库里没有图片，请先放入图片后重试")

    labels, matrix = load_library(directory, detector)
    if matrix.shape[0] == 0:
        raise RequestError("人脸库里没有检测到人脸，请先放入带人脸的图片后重试")

    return jsonify(
        {
            "ok": True,
            "mode": "query-library",
            "query": query_name,
            "query_faces": len(query_descriptors),
            "detector": detector,
            "library_images": len(images),
            "library_faces": int(matrix.shape[0]),
            "matches": rank_matches(
                query_descriptors,
                labels,
                matrix,
                "库中候选",
                top_results_from_request(),
            ),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


# =============================================================================
# Startup
# =============================================================================


def find_free_port(preferred_port: int) -> int:
    """The first free port at or after the preferred one.

    A busy 5000 should not break startup, so the next few ports are tried.
    """
    last_port = preferred_port + PORT_SEARCH_ATTEMPTS - 1
    for candidate in range(preferred_port, last_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # No SO_REUSEADDR here: on Windows it would let this probe bind a
            # port another process is already listening on.
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError(
        f"No free port between {preferred_port} and {last_port}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="人脸相似度网页版：1:N 上传比对、1:N 人脸库比对。"
    )
    parser.add_argument(
        "--dir",
        default="",
        help=f"人脸库文件夹，默认用脚本同级的 {DEFAULT_LIBRARY_FOLDER_NAME}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"起始端口，被占用时自动顺延，默认 {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--hog",
        action="store_true",
        help="网页默认选「快速」检测方式（页面上的两种方式随时可切）",
    )
    parser.add_argument(
        "--library-info",
        action="store_true",
        help="只统计人脸库有多少图多少人脸然后退出，不启动网页",
    )
    return parser


def print_library_info(detector: str) -> int:
    """The --library-info report: counts only, then exit code."""
    directory = default_library_directory()
    images = library_images(directory)
    print("正在加载模型，约 1 秒…", flush=True)
    print(f"模型加载完成，用时 {warm_up(detector):.0f} ms", flush=True)
    print(f"人脸库目录: {directory}")
    if not images:
        print(f"人脸库里没有找到图片（{' '.join(sorted(SUPPORTED_EXTENSIONS))}）")
        return 0

    _labels, matrix = load_library(directory, detector)
    print(f"图片 {len(images)} 张，检测到人脸 {matrix.shape[0]} 张")
    return 0


def serve(detector: str, preferred_port: int) -> int:
    """Warm the models up, then hand the port over to Flask."""
    print("正在加载模型，约 1 秒…", flush=True)
    print(f"模型加载完成，用时 {warm_up(detector):.0f} ms", flush=True)
    print(describe_configuration(), flush=True)
    print(f"人脸库目录: {default_library_directory()}", flush=True)

    listening_port = find_free_port(preferred_port)
    print(f"Open http://127.0.0.1:{listening_port} in your browser", flush=True)
    app.run(host="0.0.0.0", port=listening_port, debug=False, threaded=True)
    return 0


def main(argv=None) -> int:
    arguments = build_argument_parser().parse_args(argv)

    # The command line is the only configuration; nothing is read from the
    # environment, so an unrelated variable can never change this run.
    SETTINGS.library_directory = arguments.dir
    SETTINGS.default_detector = DETECTOR_HOG if arguments.hog else DETECTOR_CNN

    if arguments.library_info:
        return print_library_info(default_detector())
    return serve(default_detector(), arguments.port)


if __name__ == "__main__":
    raise SystemExit(main())