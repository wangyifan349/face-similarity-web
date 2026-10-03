# 🧑‍💻 Face Similarity Web

[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![dlib](https://img.shields.io/badge/dlib-20.0.1-blue.svg)
![InsightFace](https://img.shields.io/badge/InsightFace-2.0-blue.svg)
![Single file](https://img.shields.io/badge/architecture-single--file-lightgrey.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-black.svg)

> 🎯 One interface and one set of endpoints, two interchangeable recognition engines. This document covers both files: **`flask_face_match_v2.py`** (dlib version) and **`flask_insightface_face_v3.py`** (InsightFace version).

---

## 📖 Introduction

This program answers a question of one kind: **given a query image, find out who it looks like.** It first detects the faces in the image, extracts a fixed length feature vector for each face, converts the cosine similarity of two feature vectors into a percentage from 0 to 100, then sorts the candidates by similarity and returns a ranked list. A higher percentage means the facial features in the two photos are closer together. The program itself never concludes whether two faces are "the same person"; the actual threshold has to be calibrated by the user against their own samples.

The program offers two retrieval modes, and each can be used entirely on its own:

- 📤 **Upload comparison** — compare one query image against the candidate images uploaded in the same request. No face library is needed at all, which makes it suitable for ad hoc checks.
- 🗂️ **Library search** — compare one query image against every image in the server side face library and return the closest matches automatically, which suits looking someone up in a fixed set of people.

The two sit side by side as the left and right columns of the page, with an identical visual layout and identical interactions.

This repository provides two engine implementations. Apart from the differences listed in [Differences Between the Two Versions](#-differences-between-the-two-versions), everything else — features, endpoints and page behavior — is identical, and each one is a single file: detection, descriptors, scoring and the web service are all built in, with no second module to depend on.

---

## 🔀 Differences Between the Two Versions

| Item | dlib version | InsightFace version |
| --- | --- | --- |
| 📄 File | `flask_face_match_v2.py` | `flask_insightface_face_v3.py` |
| ⚙️ Startup configuration | Command line arguments | **No arguments**; the face library folder is asked for interactively at startup (any argument passed in is **silently ignored**) |
| 👁️ Face detection | `mmod_human_face_detector.dat` (CNN) or dlib's built-in HOG, switchable on the page | SCRFD (`det_10g.onnx`), a **single detector** |
| 🧬 Feature extraction | dlib ResNet, 128 dimensions | ArcFace, 512 dimensions |
| 📍 Landmarks model | Required (68 points, falls back to 5 points when missing) | Not required |
| 📥 Getting the weights | Download the `.bz2` files manually and unpack them | Downloaded automatically by `insightface` on first run |
| 📏 Score scale | The same person is usually above 85 | The same person is usually above 75; different people are near 0 |
| 🎛️ Detection mode dropdown on the page | Yes | No |
| ⚡ Hardware acceleration | CPU (the machine this was verified on) | CUDA is probed at startup; the GPU is used when available, otherwise it falls back to the CPU |

> ⚠️ **Scores from the two versions cannot be compared side by side.** They come from completely different feature models, and each scale is only self consistent within its own engine. See [Score Semantics](#-score-semantics).

---

## ✨ Features

| Feature | Description |
| --- | --- |
| 📦 **Single file deployment** | Each engine is compressed into one `.py` file, with no second module |
| 🔀 **Two retrieval modes** | Upload comparison and library search sit side by side and do not depend on each other |
| 🎛️ **Two detection modes** | The dlib version switches between "Accurate (CNN)" and "Fast (HOG)" with an identical scoring basis |
| 🧠 **Models stay resident** | Each model is loaded only once per process; switching modes needs no restart |
| ⚡ **Library cache** | Descriptors are cached per file and invalidated by modification time, so additions, edits and deletions take effect immediately |
| 🛡️ **Fault tolerance** | A single corrupt image, or one with no face, is skipped and logged without aborting the whole search |
| 📊 **Absolute scale visualization** | Similarity bars are drawn on a fixed scale; ⚠️ a negative score shows as an empty 0% bar while the number still displays the real signed score |
| 🔒 **No environment variables** | Neither version reads any environment variable, so the source of every setting is obvious |
| 🕵️ **The path never leaks** | The face library folder exists only on the server; neither the page nor the API exposes or accepts it |

---

## 🖥️ Interface Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Face Similarity             Detection mode [Accurate(CNN) ▾]   Library status│
├────────────────────────────────┬─────────────────────────────────────────────┤
│ 1 Query vs candidates (1:N)    │ 2 Query vs library (1:N)                    │
│                                │                                             │
│ (1) Query image (single)       │ (1) Query image (single)                    │
│ (2) Candidate images (multi)   │ Results to return                           │
│ Results to return              │                                             │
│ [ Start comparison ]           │ [ Search library ]                          │
├────────────────────────────────┴─────────────────────────────────────────────┤
│ 3 Results                                                                    │
│+--------+----------------+--------------+------------------------------+     │
│| Rank   | Candidate      | Similarity   |                              |     │
│+--------+----------------+--------------+------------------------------+     │
│| 1      | zhang.jpg      | 91.23%       | ████████████████████░░░      |     │
│| 2      | li.jpg         | 78.90%       | ██████████████░░░░░░░░       |     │
│+--------+----------------+--------------+------------------------------+     │
│ A full bar is 100%; a negative score shows no bar at all.                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Interface highlights: the two columns are equally wide, side by side and hugging the edges of the page; a thumbnail preview appears as soon as files are chosen; the button enters a loading state while a request is in flight; the closest match is highlighted in a separate summary block; errors are shown as a dismissible alert bar.

💡 The InsightFace version uses the same layout, except that no "Detection mode" dropdown appears at the top (that engine has only one detector); every other element stays in place. The third column header of the result table comes from the `kind` field of the data at hand ("Uploaded candidate" or "Library candidate").

---

## ⚙️ Requirements

Both versions share Flask, OpenCV and NumPy; the recognition engines differ in their dependencies:

| Item | Minimum | Version verified on this machine |
| --- | --- | --- |
| Python | 3.9 | 3.14.3 |
| Flask | 3.0 | 3.1.3 |
| opencv-python | 4.5 | 5.0.0.93 |
| numpy | 1.23 | 2.4.6 |

| Engine | Extra dependencies | Version verified on this machine |
| --- | --- | --- |
| dlib version | `dlib` | 20.0.1 |
| InsightFace version | `insightface`, `onnxruntime` | 2.0 / 1.30.0 |

Windows, Linux and macOS are all supported. Apart from dlib, every dependency can be installed straight from pip:

```bash
pip install flask opencv-python numpy
# then add the one matching the engine you picked:
pip install dlib                       # dlib version
pip install insightface onnxruntime   # InsightFace version
```

> 📝 The "version verified on this machine" column above lists combinations that were actually installed and verified. The code itself uses no syntax specific to anything above 3.9, so in theory the lower bound is decided by which wheels are available for `dlib` / `insightface`.

### 🔨 About Installing dlib

On some platform and interpreter combinations dlib has no prebuilt package, so a local C++ toolchain is required:

| Platform | What is needed |
| --- | --- |
| Windows | The "Desktop development with C++" workload of Visual Studio Build Tools |
| Linux | `build-essential`, `cmake` |
| macOS | Xcode Command Line Tools |

If compiling from source fails, use the prebuilt package from the conda channel instead: `conda install -c conda-forge dlib`.

---

## 🧠 Model Files (Weight Downloads)

### dlib version

Four weight files from the official dlib site are needed. **All of them are downloaded from the official dlib site; no registration and no login.**

| File | Purpose | Required | Download | Archive size |
| --- | --- | --- | --- | --- |
| `shape_predictor_68_face_landmarks.dat` | **68 point facial landmark prediction**. Once a face box has been detected, this locates the feature points that feed into feature extraction | Required (falls back to 5 points when missing) | [dlib.net/files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) | about 61 MB |
| `dlib_face_recognition_resnet_model_v1.dat` | **128 dimensional face feature extraction**. Every similarity score is computed from the feature vectors it produces | Required | [dlib.net/files](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) | about 20 MB |
| `mmod_human_face_detector.dat` | **CNN face detector** (MMOD). Only the "Accurate (CNN)" mode needs it; its recall on profile faces, small faces and cluttered backgrounds is clearly better than HOG | Optional | [dlib.net/files](http://dlib.net/files/mmod_human_face_detector.dat.bz2) | about 0.7 MB |
| `shape_predictor_5_face_landmarks.dat` | **5 point landmark prediction**, used as a fallback for the 68 point version. Enabled automatically only when the 68 point file is missing, with lower alignment accuracy | Optional | [dlib.net/files](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2) | about 5.4 MB |

📌 The program cannot work if either of the first two is missing (it raises an exception right at startup).

> ⚠️ A missing `mmod_human_face_detector.dat` is **not** degraded gracefully: the default mode is CNN, so loading the model raises an uncaught `FileNotFoundError` and the web service exits during warm up; if the service is already running, a request that switches to CNN mode returns **HTTP 500**. If you do not need CNN mode, start with `--hog`.

### InsightFace version

Two files from the `buffalo_l` model pack are needed: `det_10g.onnx` (SCRFD detection) and `w600k_r50.onnx` (ArcFace 512 dimensional features).

**No manual download is needed**: on first run `insightface` downloads the whole `buffalo_l` pack into `.insightface/models/buffalo_l` under the user directory.

The program looks for the model directory in this order; a location **only counts as a hit when both files are present in it**:

1. `insightface_models/buffalo_l/` next to the script
2. `models/buffalo_l/` next to the script
3. `buffalo_l/` next to the script
4. `~/.insightface/models/buffalo_l/` (the default download location)

💡 The first three let the models be shipped together with the program, so it runs fully offline.

📌 If a directory exists but is missing one of the files, that directory is **silently skipped** and the next one is tried; when no location is complete, the program falls back to downloading automatically. **The program never says which file is missing.**

If you are offline, run it once on a machine with network access to complete the download, then copy the whole `buffalo_l` directory to any of the locations above.

### 📂 Where to Put the dlib Weights and the Lookup Order

The program first collects every candidate path, then picks the first usable file:

1. the `models/` directory next to the script
2. the directory the script itself is in
3. any subdirectory of the script's directory (searched recursively)
4. that file name under **every** directory in `sys.path`, plus that file name under the `face_recognition_models/models/` subdirectory of each

> 📝 Step 4 **alternates between directories in `sys.path` order**, that is `sys.path[0]/<file>` → `sys.path[0]/face_recognition_models/models/<file>` → `sys.path[1]/<file>` → …, rather than scanning the whole of `sys.path` first and only then looking inside the model pack.

When no location matches, the program raises an exception right at startup and states clearly which file is missing.

🔍 **ASCII paths win**: if the candidates include a pure ASCII path, that one is picked even when a non-ASCII path came earlier in the list (because dlib cannot open non-ASCII paths). When every candidate is non-ASCII, the first one is used.

A recommended directory layout:

| Path | Purpose | Required |
| --- | --- | --- |
| `flask_face_match_v2.py` | The dlib version program itself | Required |
| `models/` | Holds the weights; put the two required ones here | Required |
| `face_library/` | The face library folder; point `--dir` somewhere else if you prefer | Optional |

### 🛍 Alternative Sources

If you would rather not download the dlib weights one by one, pick any of these:

| Route | Description |
| --- | --- |
| [face_recognition_models (PyPI)](https://pypi.org/project/face-recognition-models/) | A pip installable model pack that **contains both the 68 point landmarks file and the recognition model**; after installing it the program finds them automatically in the pack's `models` directory. **It does not contain the CNN detector**, so "Accurate (CNN)" mode still needs a separate download |
| [face_recognition_models (GitHub source)](https://github.com/ageitgey/face_recognition_models) | The source repository of that model pack, with the original weight download instructions |
| [face_recognition (GitHub)](https://github.com/ageitgey/face_recognition) | The well known Python face recognition library; its documentation lists the official addresses and the steps for getting the dlib weights above |
| [The dlib website](http://dlib.net/) | The official dlib site, describing its overall capabilities and the other models |

### 🇨🇳 About Non-ASCII Paths

dlib cannot open a path that contains non-ASCII characters directly. The dlib version has built in handling for this and mirrors such weight files into the system temp directory (`%TEMP%\dlib_face_models\`) before loading them, so no intervention is needed. The InsightFace version is built on ONNX Runtime and has no such limitation.

---

## 🚀 Running the App

### dlib version

Configured through command line arguments, reading no environment variables at all. By default it uses `face_library` next to the script as the face library folder and starts listening on port 5000; if that port is taken it moves forward automatically, trying at most 20 of them (that is 5000–5019).

| Argument | Default | Description |
| --- | --- | --- |
| `--dir` | `face_library` next to the script | Sets the face library folder; all of its subdirectories are scanned recursively |
| `--port` | `5000` | Sets the first port to listen on |
| `--hog` | Off | Makes "Fast" the default detection mode on the page; the dropdown can switch back at any time |
| `--library-info` | Off | Only counts the size of the face library, prints it and exits, without starting the web service |

```bash
python flask_face_match_v2.py --dir "D:\faces" --port 5000
python flask_face_match_v2.py --library-info          # check it over before going live
```

💡 `--library-info` is recommended for a first deployment: before serving anyone for real, it confirms that the models load correctly, that the face library path is right, and that the image and face counts are what you expected.

### InsightFace version

**Accepts no command line arguments at all** — the script does not read `sys.argv` in any way, and any argument passed in is **silently ignored** (no error, no warning), so never assume an argument took effect. The face library folder is asked for interactively at startup:

| Prompt | Behavior |
| --- | --- |
| Asked on first startup | Type the face library folder; press Enter to use `face_library` next to the script |
| The folder typed in does not exist | It says so and asks again |
| A file was typed in instead of a folder | It says so and asks again |
| Non interactive terminal | Nothing is asked; the default folder is used directly |
| The default folder does not exist | It is created automatically (`mkdir(parents=True, exist_ok=True)`) |

The models are loaded and warmed up once before startup finishes, then the reachable address is printed. The first run also downloads the `buffalo_l` pack, which takes as long as the network allows.

---

## 🔬 How It Works

The dlib version pipeline:

> 🖼️ **Input image** → decode and normalize into an RGB array → face detection → 68 point landmark location → 128 dimensional feature extraction → L2 normalization → cosine similarity converted to a percentage → sort and cut to Top-N

The InsightFace version pipeline:

> 🖼️ **Input image** → decode and normalize → SCRFD face detection (`det_size=640×640`, `det_thresh=0.5`) returning 5 point landmarks → crop and align → ArcFace 512 dimensional feature extraction → L2 normalization → cosine similarity converted to a percentage → sort and cut to Top-N

A few notes on scoring:

- 📏 The score is a **cosine similarity percentage**, that is `dot(a, b) × 100`; an image compared with itself gives 100%.
- 🔎 The **query image side** has no "take the maximum" limit: every candidate is compared against **all** faces in the query image, and the highest score becomes that candidate's score. So a group photo as the query still picks out the face that looks most like the candidate.
- 1️⃣ The **candidate image side** only uses the **largest** face for the comparison.
- 📊 The bars in the result table are clamped to the 0–100 range in the browser, so a negative score shows as an empty 0% bar while the numeric label still displays the real signed score.

> 📝 Both files export a `face_similarity_percent()` library function whose multi-face default is **exactly the opposite** in the signature: the dlib version defaults to `compare_all_faces=False` (take the largest face), the InsightFace version to `largest_only=False` (take the best pairing). That function is **not called by any HTTP route**, and the web behavior follows the rules above.

---

## 📏 Score Semantics

The two engines use different feature models, so scores are **only comparable within each engine**; comparing across engines is meaningless.

| Sample | dlib version | InsightFace version |
| --- | --- | --- |
| The same image compared with itself | 100.00% | 100.00% |
| Same person, different photos (frontal crop vs half body shot) | 96.85% | 77.57% |
| Different people (several samples) | clearly below the same person | −3% to +1% |

The table above comes from a single set of samples measured on one machine and serves only to illustrate the difference in magnitude. ArcFace features from InsightFace separate different people more sharply, which is why the same person scores lower than with dlib while different people score near zero; this does not mean either version is more accurate, only that the two scales differ.

💡 Recommended practice: once you have picked an engine, calibrate the threshold with several same-person and different-person samples from your own scenario. Do not copy the numbers in the table above directly, and do not put scores from the two engines on the same leaderboard.

---

## 🎛️ Detection Modes (dlib version)

| Item | Accurate (CNN) | Fast (HOG) |
| --- | --- | --- |
| 👁️ Detector | `mmod_human_face_detector.dat` | dlib's built-in one, no weights file needed |
| 🔍 Upsampling passes | 0 | 1 |
| 🧩 Best suited for | Profile faces, small faces, cluttered backgrounds | Frontal faces, even lighting |
| ⏱️ Relative cost | High | Low |
| 📍 Landmarks model | The same | The same |
| 🧬 Feature model | The same | The same |
| 📊 Scoring logic | The same | The same |

Each extra level of upsampling multiplies the time by about four while also improving recall on small faces; hence CNN uses 0 and HOG uses 1. The two modes **differ only in the face detection step**; the landmark, feature and scoring chain after it is exactly the same, so the scores the two modes produce can be compared directly. Each mode caches its own descriptors (the cache key includes the detection mode), so switching back and forth never re-encodes the library.

The InsightFace version has only the SCRFD detector and offers no such option.

---

## 🗂️ Face Library Conventions

- 📍 **Location**: in the dlib version it is set by `--dir` or the default folder; in the InsightFace version by the answer given at startup or the default folder. Both walk all subdirectories recursively.
- 🖼️ **Formats**: `.jpg` `.jpeg` `.png` `.bmp` `.webp` are supported, with the extension matched case insensitively.
- 👥 **Files with several faces (library search)**: **every** face in one image becomes its own candidate, with `#2`, `#3` and so on appended from the second one on.
- 📤 **Files with several faces (upload comparison)**: only the **largest** face of each uploaded image is used; no suffix is appended, and one image with several faces never splits into several candidates.
- 🏷️ **Duplicate file names**: files of the same name in different folders are shown as "name (1)", "name (2)" in the **upload comparison** results; ⚠️ that de-duplication **does not apply to library search** — a `zhang.jpg` in two different subfolders shows as `zhang.jpg` in both places during a library search.
- ⚡ **Cache policy**: the cache key is the absolute path of the file and validity is decided by its modification time; edits, additions and deletions of images are all reflected on the next search, and the cache entries of deleted files are cleaned up at the same time. The dlib version additionally counts the detection mode in the cache key.
- 🛡️ **Fault tolerance**: when one image cannot be read or holds no face, that entry is skipped; upload comparison lists the reason in the `skipped` field, while library search skips it silently.
- 📊 **Progress output**: once the face library holds **20 images or more**, both versions print per image load progress in the terminal (cache hits are printed too).
- 🏠 **Folder creation**: the InsightFace version creates the default face library folder automatically at startup. The dlib version does not create folders; prepare one yourself.

---

## ⏱️ Performance Reference

The figures below were measured on this machine (AMD Ryzen 7 5800H / 60 GB RAM / Windows / Python 3.14.3). They are for reference only and **are not a performance guarantee**.

dlib version, with 640×480 input:

| Operation | Time |
| --- | --- |
| Model loading (HOG mode) | about 1255 ms |
| Model loading (CNN mode) | about 1049 ms |
| HOG face detection (640×480) | about 142 ms |
| CNN face detection (640×480) | about 4625 ms |
| Landmarks + 128 dimensional feature extraction (one face) | about 287 ms |
| Similarity computation for 500 candidates × 1 query | about 0.5 ms |

InsightFace version (the execution provider reported at startup on this machine was CPUExecutionProvider):

| Operation | Time |
| --- | --- |
| Model loading + first warm up (including ONNX session setup) | about 1.7 s |
| 1 query × 3 library images end to end (1600×1600 large images, first run includes encoding) | about 149 – 676 ms |
| The same, after the cache is hit | single digit milliseconds |

A few conclusions and suggestions:

- 🎯 In both engines the performance bottleneck sits in **face detection**; the cost of feature extraction and similarity computation is negligible.
- ⚡ The first search of a large face library takes noticeably longer; later requests hit the cache and are usually single digit milliseconds. The cache lives in process memory and is lost on restart.
- 💡 For the dlib version, use HOG mode as the daily default and switch to CNN when needed; CNN detection time rises quickly with image resolution, so normalizing image sizes before they enter the library pays off clearly.
- 💡 The InsightFace version needs no detector choice; SCRFD is fairly sensitive to input resolution, so normalizing sizes before ingest is recommended here too.

> ⚠️ These numbers vary with hardware, image size and image content; run your own benchmark in the target environment before deploying. The individual InsightFace stages were not measured separately, one by one.

---

## 🔌 HTTP API

The routes and response structure are identical in both versions. Every endpoint is a stateless call. Success returns HTTP 200; failure returns a body containing `ok: false` and an `error` description, with 400 as the default status code and 413 when the request body exceeds 64 MB.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Returns the web page |
| GET | `/api/library` | Size statistics of the face library, **returning counts only, never the folder path** |
| POST | `/api/query-set` | Compares the query image with the candidate images uploaded in the same request (1:N) |
| POST | `/api/query-library` | Compares the query image with the whole face library (1:N) |

### 🔧 Common Parameters

| Parameter | Location | Accepted values | Default | Applies to |
| --- | --- | --- | --- | --- |
| `top_results` | Form or query string | Integer, range 1–100 | 10 | Both versions |
| `detector` | Form or query string | `cnn` or `hog` | `cnn` | dlib version only |

📌 The rules for `top_results`: **missing or not a number → 10**; parsable as an integer → **clamped to the bounds** 1–100 (`0` → 1, `999` → 100).

> ⚠️ The two versions differ slightly here: the dlib version parses with `int()` (which accepts `+5`, surrounding spaces and the like), while the InsightFace version tests with `str.isdigit()` first and only then converts (rejecting `+5`, a minus sign and Unicode digits). Given `top_results=-1`, the dlib version returns 1 row and the InsightFace version returns 10. Requests made normally from the page are unaffected.

> ⚠️ `GET /api/library` in the dlib version reads `detector` from the query string, so **the same URL returns a different `faces` count under different detection modes**; the InsightFace version ignores the parameter.

### 📤 Upload Comparison (`/api/query-set`) Form Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | File | Yes | The query image, a single one |
| `candidates` | File (repeatable) | Yes | The candidate images, one or more |

### 🗂️ Library Search (`/api/query-library`) Form Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | File | Yes | The query image, a single one |

### ✅ Success Response Fields

| Field | Description |
| --- | --- |
| `ok` | Whether the call succeeded |
| `mode` | The retrieval mode, `query-set` or `query-library` |
| `query` | The file name of the query image |
| `query_faces` | The number of faces detected in the query image |
| `detector` | The detection mode actually used (**dlib version only**) |
| `count` | The total number of candidate images submitted (**upload comparison only**) |
| `compared` | The number of candidate images that were successfully encoded and took part in the comparison (**upload comparison only**) |
| `library_images` | The total number of images in the face library (**library search only**) |
| `library_faces` | The total number of faces detected in the face library (**library search only**) |
| `matches` | The ranked result list, holding `rank`, `kind`, `candidate` and `similarity`, where the first item also carries `best: true` |
| `skipped` | The candidate images that were skipped, with the reason (**upload comparison only**) |
| `elapsed_ms` | Server side processing time in milliseconds |

The values of the `kind` field: `query-set` gives "Uploaded candidate", `query-library` gives "Library candidate".

### ❌ Common Errors

| Status code | Situation |
| --- | --- |
| 400 | No query image chosen, more than one query image, no candidate images chosen, no usable face among the candidates, the face library is empty, no face found in the library |
| 413 | The request body exceeds 64 MB |
| 500 | The dlib version is missing the CNN detector weights but is still asked to handle a request in CNN mode |

---

## 🔒 Security and Privacy

- ⚠️ **This project does not produce an identity verdict.** It only outputs similarity numbers; choosing the "same person or not" threshold, and the misjudgments that follow from it, are the user's responsibility.
- 🌐 **The service listens on `0.0.0.0` by default**, so any other host on the same LAN can reach it. To restrict it to this machine, change `host` to `127.0.0.1` in the `app.run(...)` call at the end of the file. ⚠️ Note: the address printed in the terminal is always `http://127.0.0.1:<port>`, even when every network interface is actually being listened on — do not use it to judge the exposure.
- 🧠 **Uploaded images never touch the disk**: they are decoded and computed in memory only and released when the process exits. Face library images, on the other hand, are read and cached in process memory.
- 🕵️ **The face library path never leaks**: neither the page nor the API displays or accepts it; the path exists only inside the server process (`/api/library` returns counts only).
- 🕳️ **There is no authentication**: the program ships without identity verification or access control. If you deploy it on an untrusted network, be sure to add authentication in a reverse proxy layer.
- 🐌 **Do not use a network share as the face library**: heavy small file network I/O noticeably degrades performance.
- ⚠️ **The built-in server is the Flask development server**, suitable only for small scale use inside a private network; in production put it behind a WSGI server such as Gunicorn or uWSGI.

---

## ⚠️ Known Limitations

- 🚫 No face alignment enhancements such as cropping or rotation correction of detection results: the dlib version uses only the default landmark prediction, and the InsightFace version uses the 5 point alignment that comes with SCRFD.
- 💾 The descriptor cache lives in process memory, so a restart means re-encoding the whole library.
- 📏 The descriptors come from dlib ResNet (128 dimensions) and ArcFace (512 dimensions) respectively, so their scores are **not directly comparable** with those of other face recognition solutions; nor are the two versions comparable with each other.
- 👥 In upload comparison only the largest face of each candidate image is used, so the other people in a group photo never become candidates of their own (library search does expand every face).
- 🚫 There is no batch offline comparison endpoint and no result persistence.
- ⚡ Hardware acceleration cannot be specified by hand: the dlib version ran on the CPU in the verification environment; the InsightFace version probes `CUDAExecutionProvider` at startup and prefers the GPU when available, otherwise falling back to the CPU. There is no argument to force an execution device.
- 🐛 The InsightFace version handles invalid Unicode digits in `top_results` (such as `²`) badly and may raise an uncaught exception.

---

## ❓ FAQ

**❌ The dlib version cannot find the `.dat` weight files at startup**
The models are not in a place the program can search. See the [Model Files](#-model-files-weight-downloads) section for where to put them, or install the `face_recognition_models` pack to get the two required weights.

**❌ The InsightFace version downloaded the models again after startup**
That means none of the three `buffalo_l` folders next to the script, nor `~/.insightface/models/buffalo_l`, holds both `det_10g.onnx` and `w600k_r50.onnx` at once — incomplete folders are skipped silently. Check that the files are complete, or run it with `--` and no other argument to watch the startup log.

**❌ It says "The face library has no images; add some images first and try again"**
The face library folder does not exist, or holds no supported image format. In the dlib version check with `--library-info` first; in the InsightFace version restart and type the right folder when prompted.

**❌ It says "query image no face detected: ..."**
The current detector found no face in the query image. In the dlib version switch to "Accurate (CNN)" mode and try again — CNN is more forgiving with profile faces and small faces; also make sure you uploaded the original photo rather than a small crop taken from a screenshot.

**❌ It says "No face detected in the face library; ..."**
The library holds images but none of them yielded a face. Check the actual detected count first, and spot check the sharpness and size of the images.

**❓ The first search is slow and later ones are fast**
That is the expected behavior. The first run has to encode the whole library; once the results are cached, later requests are single digit milliseconds.

**❓ The port is already in use**
Both versions move forward automatically, at most 20 ports (5000–5019). Always go by the address actually printed in the terminal.

**❓ I replaced an image but the search results did not change**
That should not happen. The cache is decided by the file modification time, so check that you modified the file at that same path in the library.

**❓ Two images of the same name in the library both show as `zhang.jpg`**
That is how the current implementation works. The "name (1)", "name (2)" de-duplication applies to the **upload comparison** results only; library search does not de-duplicate.

**❓ Can the weights live under a path with non-ASCII characters in it**
Yes. The dlib version mirrors the model into the system temp directory before loading it; the InsightFace version has no such limitation. The program also prefers a pure ASCII path among several candidates.

**❓ Why does a candidate have no score / not appear in the results**
The image could not be read, or no face was detected in it. In the upload comparison response the `skipped` field lists the reason; library search skips silently, and `--library-info` lets you compare the image count with the face count.

**❓ Can scores from the two versions be compared together**
No. See [Score Semantics](#-score-semantics); the scales differ, so calibrate each threshold separately.

**❓ A colleague on the LAN cannot open this page**
Both versions bind `0.0.0.0` by default, so in theory it is reachable; if the connection fails, check whether the system firewall allows that port.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0**. The full text is in [`LICENSE`](./LICENSE) or at <https://www.gnu.org/licenses/agpl-3.0.html>.

```
Copyright (C) 2026 Alex Walker
```

**⚠️ Special note on AGPL-3.0**: if you offer this program as a running service to users over a network, you **must** also give those users the complete corresponding source code of the program (including your modifications to it). This is the core requirement AGPL-3.0 adds on top of GPL.

This project is released "as is", with no warranty of any kind, express or implied. The authors and copyright holders are not liable for any direct or indirect loss arising from the use of this program.