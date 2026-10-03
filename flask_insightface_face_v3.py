"""人脸相似度网页版（InsightFace）。

启动后会先问你人脸图片放在哪个文件夹，问完就把网页服务跑起来：打开页面，
选一张查询图，再选一组候选图，就能看到每张候选图跟查询图有多像；或者让
查询图去和你的人脸文件夹比。人脸由 SCRFD 检测、ArcFace 描述，每张脸变成
512 维特征（做过 L2 归一化），得分是两张脸特征的余弦相似度乘以 100。同一个
人的不同照片实测 77-79，不同的人 -3 到 +1。这个程序不替你判定"是不是同一
个人"，阈值要拿自己的样本定。

分数来自 ArcFace，和 dlib 版（flask_face_match_v2.py）不是同一个模型，数值
不能互相比较。网页的布局、配色和那条固定 0-100 的相似度条跟 dlib 版一样，
只有两处不同：没有检测方式下拉框（SCRFD 是唯一检测器），结果里也没有
detector 字段。

装依赖（Python 3.9 及以上）：

    pip install flask opencv-python numpy insightface onnxruntime

有 N 卡想用 GPU，把 onnxruntime 换成 onnxruntime-gpu。

跑起来：

    python flask_insightface_face_v3.py

然后照提示输入人脸库文件夹，直接回车就用脚本同级的 face_library。图片支持
.jpg .jpeg .png .bmp .webp，可以有子目录。页面上不显示也不接受文件夹路径，
所以这个位置只有你启动时知道。

buffalo_l 模型（约 300 MB）在首次运行时自动下载到 ~/.insightface/models/
buffalo_l；脚本同级已经有 insightface_models/buffalo_l 或 models/buffalo_l 时
就地使用，不联网。

所有配置都写在代码里或运行时问你要，不读任何环境变量。
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

# 只加载 buffalo_l 里的检测和识别两个模型。关键点和性别年龄模型用不上，
# 加载它们只会拖慢启动。
ALLOWED_MODULES = ["detection", "recognition"]
REQUIRED_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
MODEL_FOLDER = Path("~/.insightface").expanduser() / "models" / "buffalo_l"
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# InsightFace 的关键点模型调用了 scikit-image 的旧接口，每检测到一张脸就警告
# 一次。做相似度不需要这个提示，所以在这里统一关掉一次。
warnings.filterwarnings("ignore", message=r".*estimate.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.*")

# 人脸库文件夹，由启动时的询问填进来；之后一直用它。
LIBRARY_FOLDER = SCRIPT_DIRECTORY / "face_library"

# 已加载的 FaceAnalysis。构造它要两秒多，所以只做一次，之后每个请求都用它。
FACE_APP = None
MODEL_SUMMARY = ""

# 文件夹 -> (修改时间, 该文件里每张脸的特征)
LIBRARY_CACHE = {}
LIBRARY_LOCK = threading.Lock()


# =============================================================================
# 找模型、加载模型
# =============================================================================


def find_model_directory() -> Optional[Path]:
    """已经在磁盘上的 buffalo_l 目录，找不到就返回 None。

    脚本同级优先，这样把模型放在程序旁边就能离线跑；实在没有才让
    InsightFace 去下载。
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
    """加载检测和识别模型，返回一行给日志看的说明。

    只在这个进程里执行一次，之后的请求直接用 FACE_APP。
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
        # 让 InsightFace 自己去 MODEL_FOLDER 下载。
        FACE_APP = FaceAnalysis(
            name="buffalo_l",
            root=str(MODEL_FOLDER.parents[1]),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = f"{MODEL_FOLDER}（本次下载）"
    else:
        FACE_APP = FaceAnalysis(
            name=str(found),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = str(found)

    # ctx_id 只有在 CUDA 后端下才有意义。
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    FACE_APP.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)

    MODEL_SUMMARY = (
        f"model=buffalo_l dim=512 providers={','.join(providers)} models={where}"
    )
    return MODEL_SUMMARY


def get_app():
    """已经加载好的模型，必要时先加载。"""
    if FACE_APP is None:
        load_model()
    return FACE_APP


# =============================================================================
# 把各种输入统一成 BGR 数组
#
# 往下的代码只认一种形状：连续的 BGR uint8 数组，OpenCV 和 InsightFace 都
# 要这个格式。
# =============================================================================

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def to_bgr_array(image: ImageInput, input_is_bgr: bool = True) -> np.ndarray:
    """把路径、字节或数组统一成连续的 BGR uint8 数组。

    input_is_bgr=True 表示传进来的数组已经是 BGR 顺序（OpenCV 自己就是这个
    顺序，默认值）；传 False 表示它是 RGB，这里负责翻成 BGR。给路径或字节时
    这个参数不起作用。
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            array = np.repeat(image[:, :, None], 3, axis=2)      # 灰度变三通道
        elif input_is_bgr:
            array = image
        else:
            array = image[:, :, ::-1]                              # RGB 变 BGR
        if array.dtype != np.uint8:
            # 另一种常见约定是 [0, 1] 的浮点数组。
            if np.issubdtype(array.dtype, np.floating) and array.size > 0:
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype("uint8")
        return np.ascontiguousarray(array)

    # cv2.imread 读不了 Windows 上的中文路径，所以自己读字节再解码，
    # 所有输入类型都走这一条路。
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        data = image_path.read_bytes()

    decoded = cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("图片无法解码，可能已损坏或不是图片")
    return np.ascontiguousarray(decoded)


# =============================================================================
# 提特征、算分数
#
# 这一段是整个引擎。网页那部分只负责搬图片和搬 JSON，别的什么都不做。
# =============================================================================


def _face_box_area(face) -> float:
    """检测框的面积，单位像素。"""
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
    """一个特征向量除以自己的长度。全零向量返回 None。"""
    array = np.asarray(vector, dtype="float32").reshape(-1)
    length = float(np.linalg.norm(array))
    if length == 0.0:
        return None
    return array / length


def encode_faces(image: ImageInput, input_is_bgr: bool = True) -> list:
    """图片里每张脸一个 512 维特征，已经 L2 归一化。

    按人脸框面积从大到小排好，所以第 0 个就是照片里最大的那张脸，调用方
    不用再关心一张图里有几张脸。没有检测到人脸就返回空列表。
    """
    detected = encode_faces_with_scores(image, input_is_bgr)
    embeddings = []
    for face in detected:
        embeddings.append(face["embedding"])
    return embeddings


def encode_faces_with_scores(
    image: ImageInput, input_is_bgr: bool = True
) -> list:
    """和 encode_faces 一样，额外带上检测器自己的置信度和框面积。

    每项是 {"embedding": 特征, "det_score": 检测置信度, "area": 框面积}。
    想知道检测器有多确定时才需要用这个。
    """
    faces = get_app().get(to_bgr_array(image, input_is_bgr))

    results = []
    for face in faces:
        # 这里不能写 `or []`：特征是 NumPy 数组，对数组判断真假会报
        # "truth value of an array with more than one element is ambiguous"。
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

    # 大的在前，调用方可以直接拿第 0 个当"这张照片的人"。
    results.sort(key=lambda row: row["area"], reverse=True)
    return results


def cosine_similarity_percent(first: np.ndarray, second: np.ndarray) -> float:
    """两个特征的余弦相似度，换算成百分数。"""
    left = np.asarray(first, dtype="float32").reshape(-1)
    right = np.asarray(second, dtype="float32").reshape(-1)
    if left.size != right.size:
        raise ValueError(f"特征长度不一致：{left.size} 和 {right.size}")
    return float(np.dot(left, right)) * 100.0


def best_pair_percent(first_faces: Sequence, second_faces: Sequence,
                      largest_only: bool = False) -> float:
    """两张照片之间最高的那个分数。

    合影默认就能对上，不用管每边有几张脸。largest_only=True 时只比每边最大
    的那张脸。
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
    """每个候选各出一个最高分，按候选原来的顺序排好。

    查询图可能有多张脸（合影），所以每个候选都去和查询图里最合的那张脸比。
    用一次矩阵乘法算完，而不是在 Python 里一层层循环，人脸库大的时候差别
    很明显。
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
    """两张图片之间的人脸相似度，百分数。

    input_is_bgr 的意思见 to_bgr_array。largest_only=True 时每张图只取最大
    的那张脸来比；默认是所有脸两两比，取最高分，合影不用先裁脸。

    有一张图里检测不到人脸就返回 None。分数越高越像，实测同一个人不同照片
    77-79，不同的人 -3 到 +1。
    """
    faces_a = encode_faces(image_a, input_is_bgr)
    faces_b = encode_faces(image_b, input_is_bgr)
    if not faces_a or not faces_b:
        return None

    score = best_pair_percent(faces_a, faces_b, largest_only)
    # 浮点误差可能让结果稍微越界，压回 -100 到 100。
    if score > 100.0:
        return 100.0
    if score < -100.0:
        return -100.0
    return score


# =============================================================================
# 人脸库
#
# 人脸库就是任何一个装图片的文件夹。整个库算一遍要花不少时间，所以每个文件
# 只算一次，缓存起来，并且用文件的修改时间判断有没有变：改了、加了、删了图，
# 下一次比对就能看出来。
# =============================================================================


def library_images() -> list:
    """人脸库文件夹里所有图片，含子目录，顺序固定。"""
    found = []
    if not LIBRARY_FOLDER.is_dir():
        return found
    for path in LIBRARY_FOLDER.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    found.sort()
    return found


def _embeddings_of(path: Path) -> list:
    """一个文件里所有脸的特征，文件没变就直接用缓存。

    读不了或者里面没有人脸的，返回空列表而不是报错：一张坏图不该把整个
    人脸库拖垮。
    """
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        return []          # 刚扫描完就被删掉了

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
    """人脸库里每一张脸，返回 (名字列表, 特征矩阵)。

    名字就是文件名；一张照片里有好几张脸时后面加 " #2"、" #3"。特征矩阵
    一行对应一个名字，顺序一致。
    """
    images = library_images()
    labels = []
    embeddings = []
    alive = []

    # 一次只让一个人扫库：ONNX 会话和检测网络都怕两个请求挤在一起。
    with LIBRARY_LOCK:
        for index, path in enumerate(images, start=1):
            if len(images) >= 20:
                print(f"  载入人脸库 {index}/{len(images)}  {path.name}", flush=True)

            alive.append(str(path))
            position = 0
            for embedding in _embeddings_of(path):
                position += 1
                if position == 1:
                    labels.append(path.name)
                else:
                    labels.append(f"{path.name} #{position}")
                embeddings.append(embedding)

        # 从人脸库里删掉的图，缓存也一起清掉。
        for key in list(LIBRARY_CACHE):
            if key not in alive:
                del LIBRARY_CACHE[key]

    if not embeddings:
        return labels, np.zeros((0, 512), dtype="float32")
    return labels, np.vstack(embeddings).astype("float32")


# =============================================================================
# 分数变成结果表
# =============================================================================


def rank_matches(
    query_faces: list,
    labels: list,
    candidate_faces: Sequence,
    group_label: str,
    top_results: int,
) -> list:
    """每个候选都跟查询图比一遍，只留下前几名。

    labels 是每个候选显示用的名字，顺序和 candidate_faces 一致。group_label
    说明候选来自哪里，会作为每行的 "kind" 显示出来（上传的是"候选图片"，
    人脸库里的是"库中候选"）。

    返回的行是 {"rank", "kind", "candidate", "similarity"}，第一名多一个
    "best": True。没有候选就返回空列表。
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
    """重名的文件后面加上序号，免得结果表里两行长得一模一样。"""
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
# 网页
# =============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


class RequestError(Exception):
    """这次请求有问题，消息是给用户看的。"""


def error_response(message: str, status: int = 400):
    """页面和直接调接口的人看到的同一种出错格式。"""
    return jsonify({"ok": False, "error": message}), status


@app.errorhandler(RequestError)
def handle_request_error(error: RequestError):
    return error_response(str(error))


@app.errorhandler(413)
def handle_too_large(_error):
    return error_response("上传文件太大，单次请求上限 64 MB", 413)


def read_uploads(field_name: str) -> list:
    """某个字段里上传的所有文件，(文件名, 字节内容)。"""
    uploads = []
    for item in request.files.getlist(field_name):
        if item and item.filename:
            uploads.append((item.filename, item.read()))
    return uploads


def encode_upload(data: bytes, label: str, file_name: str) -> list:
    """一张上传图片里所有脸的特征。

    图片不能用（解码失败、损坏、里面没有人脸）就直接报 RequestError，消息
    里带上是哪张图，label 是"查询图片"或"候选图片"。
    """
    try:
        faces = encode_faces(data)
    except (OSError, ValueError) as error:
        raise RequestError(f"{label}处理失败：{error}：{file_name}")
    if not faces:
        raise RequestError(f"{label}未检测到人脸：{file_name}")
    return faces


def query_from_request() -> tuple:
    """这次请求的查询图，(文件名, 所有脸的特征)。"""
    uploads = read_uploads("query")
    if not uploads:
        raise RequestError("请选择一张查询图片")
    if len(uploads) > 1:
        raise RequestError("查询图片只能选一张")

    name, data = uploads[0]
    return name, encode_upload(data, "查询图片", name)


def top_results_from_request() -> int:
    """要返回前几名，超出范围的按边界算。"""
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


# -------------------- 页面 --------------------
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
</style>
</head>
<body>
<div class="container-fluid px-2 py-3">

  <div class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-3 px-2">
    <div>
      <h1 class="h3 mb-1" style="color:#7a2e0a">人脸相似度</h1>
      <div class="text-secondary">上传比对 · 人脸库检索</div>
    </div>
    <div class="text-end">
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
    postJson(url, new FormData(form), $(buttonId), idleText);
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
  return fetch("/api/library")
    .then((response) => response.json())
    .then((payload) => {
      $("libraryBadge").textContent = payload.ok
        ? "人脸库 " + payload.faces + " 张人脸 / " + payload.images + " 张图"
        : "人脸库不可用";
    })
    .catch(() => { $("libraryBadge").textContent = "人脸库状态未知"; });
}

loadLibraryStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """页面本身，两个面板都在上面。这页没有检测方式下拉框。

    三个数字是页面上"显示前几名"那个输入框的初值和上下限，和
    top_results_from_request 里的边界保持一致。
    """
    return render_template_string(PAGE, top_results=10, top_results_min=1, top_results_max=100)


@app.get("/api/library")
def api_library():
    """人脸库里有多少张图、多少张脸。

    只回数字，文件夹路径本身永远不离开服务器。
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
    """面板一：查询图跟这次一起上传的候选图比。"""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    uploads = read_uploads("candidates")
    if not uploads:
        raise RequestError("请至少选择一张候选图片")

    # 有一张图不能用就记下来跳过，不影响整次比对。
    labels = []
    candidate_faces = []
    skipped = []
    for name, data in uploads:
        try:
            faces = encode_upload(data, "候选图片", name)
        except RequestError as error:
            skipped.append({"name": name, "status": str(error)})
            continue
        labels.append(name)
        candidate_faces.append(faces[0])        # 这张照片里最大的那张脸

    if not candidate_faces:
        raise RequestError("候选图片里没有可用的人脸")

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
                "候选图片",
                top_results_from_request(),
            ),
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.post("/api/query-library")
def api_query_library():
    """面板二：查询图跟人脸库里每一张图比。"""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    if not library_images():
        raise RequestError("人脸库里没有图片，请先放入图片后重试")

    labels, embedding_matrix = load_library()
    if embedding_matrix.shape[0] == 0:
        raise RequestError("人脸库里没有检测到人脸，请先放入带人脸的图片后重试")

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
                "库中候选",
                top_results_from_request(),
            ),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


# =============================================================================
# 启动
# =============================================================================


def ask_library_folder() -> Path:
    """启动时问一句人脸库在哪儿，直接回车就用脚本同级的 face_library。

    没有可问的窗口时（比如双击运行或者被别的程序拉起来）不问，直接用默认
    位置，不去读一个已经关掉的输入流。
    """
    default = SCRIPT_DIRECTORY / "face_library"
    if not sys.stdin.isatty():
        default.mkdir(parents=True, exist_ok=True)
        return default

    print("人脸库就是放图片的文件夹，可以有子目录。")
    print(f"支持的格式：{' '.join(sorted(SUPPORTED_EXTENSIONS))}")
    while True:
        answer = input(f"人脸库文件夹（直接回车用 {default}）：").strip()
        if not answer:
            default.mkdir(parents=True, exist_ok=True)
            return default
        folder = Path(answer).expanduser()
        if folder.is_dir():
            return folder
        if folder.exists():
            print("那是一个文件，不是文件夹，请再输入一次。")
        else:
            print("这个文件夹不存在，请检查路径后再输入一次。")


def choose_port(preferred: int = 5000) -> int:
    """从 preferred 往后找一个没人占的端口。

    5000 经常被别人占着，占了就自动往后挪，免得启动失败。
    """
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # 这里不设 SO_REUSEADDR：Windows 上设了会让这个探测占到别的
            # 程序正在监听的端口。
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"{preferred} 到 {preferred + 19} 之间没有空闲端口")


def main():
    """问清楚人脸库在哪儿，加载模型，然后把网页服务跑起来。"""
    global LIBRARY_FOLDER
    LIBRARY_FOLDER = ask_library_folder()

    print("正在加载 buffalo_l 模型，约 2 秒…", flush=True)
    load_model()
    print("模型加载完成")
    print(MODEL_SUMMARY, flush=True)
    print(f"人脸库目录: {LIBRARY_FOLDER}", flush=True)

    images = library_images()
    print(f"人脸库里现在有 {len(images)} 张图片", flush=True)

    port = choose_port()
    print(f"Open http://127.0.0.1:{port} in your browser", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()