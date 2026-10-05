"""얼굴 유사도 웹 버전（InsightFace）。

시작하면 먼저 얼굴 사진이 어느 폴더에 있는지 묻고, 답을 받은 다음 웹 서비스를 띄웁니다.
페이지를 열어 질의 이미지를 한 장 고르고 후보 이미지 묶음을 고르면, 각 후보 이미지가
질의 이미지와 얼마나 비슷한지 볼 수 있습니다. 또는 질의 이미지를 직접 얼굴 폴더와 비교할 수도 있습니다. 얼굴은 SCRFD로 검출하고 ArcFace로 기술하며, 얼굴 하나마다
512차원 특징（L2 정규화 완료）이 되고, 점수는 두 얼굴 특징의 코사인 유사도에 100을 곱한 값입니다. 같은
사람의 서로 다른 사진은 실측 기준 77~79, 다른 사람은 -3~+1입니다. 이 프로그램은 "같은
사람인지"를 대신 판정해 주지 않으므로, 임계값은 직접 준비한 샘플로 정해야 합니다.

점수는 ArcFace에서 나오며 dlib판（flask_face_match_v2.py）과 다른 모델이므로 수치를
서로 비교할 수 없습니다. 웹 페이지의 레이아웃과 색상, 그리고 0~100으로 고정된 유사도
막대는 dlib판과 같으며, 두 곳만 다릅니다. 검출 방식 드롭다운이 없고（SCRFD만 사용）,
결과에도 detector 필드가 없습니다.

의존성 설치（Python 3.9 이상）：

    pip install flask opencv-python numpy insightface onnxruntime

N카드가 있어 GPU를 쓰려면 onnxruntime를 onnxruntime-gpu로 바꾸세요.

실행하기：

    python flask_insightface_face_v3.py

그런 다음 안내에 따라 얼굴 라이브러리 폴더를 입력하세요. 그냥 Enter를 누르면 스크립트와
같은 위치의 face_library를 씁니다. 이미지는 .jpg .jpeg .png .bmp .webp를 지원하며,
하위 폴더가 있어도 됩니다. 페이지에는 폴더 경로를 표시하지 않고 입력받지도 않으므로,

이 위치는 실행할 때만 알 수 있습니다.
buffalo_l 모델(약 300 MB)은 첫 실행 때 자동으로 ~/.insightface/models/buffalo_l에
내려받습니다. 스크립트와 같은 위치에 이미 insightface_models/buffalo_l이나

models/buffalo_l이 있으면 그 자리에서 그대로 쓰며, 네트워크에 접속하지 않습니다.
"""

import socket
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import quote

import cv2
import numpy as np
import onnxruntime
from flask import (
    Flask,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)

# buffalo_l에서 검출과 인식 두 모델만 읽어옵니다. 랜드마크와 성별·나이 모델은 쓰이지
# 않으므로, 읽어들이기만 하면 시작이 느려질 뿐입니다.
ALLOWED_MODULES = ["detection", "recognition"]
REQUIRED_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
MODEL_FOLDER = Path("~/.insightface").expanduser() / "models" / "buffalo_l"
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# InsightFace의 랜드마크 모델이 scikit-image의 구버전 API를 호출해 얼굴 하나를 찾을 때마다
# 경고를 한 번 냅니다. 유사도 계산에는 이 알림이 필요 없으므로 여기서 한꺼번에 끕니다.
warnings.filterwarnings("ignore", message=r".*estimate.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.*")

# 얼굴 라이브러리 폴더. 시작할 때 물어본 답으로 채워지고 이후 계속 쓰입니다.
LIBRARY_FOLDER = SCRIPT_DIRECTORY / "face_library"

# 결과 표의 각 줄에 썸네일을 보여주려면 브라우저가 그 이미지를 다시 받아야 합니다. 폴더가 아니라
# 라우트만 공개합니다. 요청에 적을 수 있는 것은 얼굴 라이브러리 내부의 상대 경로뿐입니다.
LIBRARY_IMAGE_ROUTE = "/library-image"
LIBRARY_IMAGE_DOWNLOAD_ROUTE = "/library-image-download"

# 이미 로드된 FaceAnalysis. 생성에 2초 넘게 걸리므로 한 번만 만들고 이후 모든 요청이 공유합니다.
FACE_APP = None
MODEL_SUMMARY = ""

# 폴더 -> (수정 시각, 그 파일에 있는 얼굴마다의 특징)
LIBRARY_CACHE = {}
LIBRARY_LOCK = threading.Lock()


# =============================================================================
# 모델 찾기, 모델 로드
# =============================================================================


def find_model_directory() -> Optional[Path]:
    """디스크에 이미 있는 buffalo_l 디렉터리. 없으면 None을 반환합니다.

    스크립트와 같은 위치를 먼저 쓰므로, 모델을 프로그램 옆에 두면 오프라인으로 실행할 수
    있습니다. 그것도 없을 때만 InsightFace가 다운로드하게 합니다.
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
    """검출·인식 모델을 읽어오고, 로그용 한 줄 설명을 반환합니다.

    이 프로세스에서 한 번만 실행되고, 이후 요청은 FACE_APP를 그대로 씁니다.
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
        # InsightFace가 MODEL_FOLDER로 직접 다운로드하게 합니다.
        FACE_APP = FaceAnalysis(
            name="buffalo_l",
            root=str(MODEL_FOLDER.parents[1]),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = f"{MODEL_FOLDER}（이번에 다운로드됨）"
    else:
        FACE_APP = FaceAnalysis(
            name=str(found),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = str(found)

    # ctx_id는 CUDA 백엔드에서만 의미를 가집니다.
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    FACE_APP.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)

    MODEL_SUMMARY = (
        f"model=buffalo_l dim=512 providers={','.join(providers)} models={where}"
    )
    return MODEL_SUMMARY


def get_app():
    """이미 로드된 모델. 필요하면 먼저 로드합니다."""
    if FACE_APP is None:
        load_model()
    return FACE_APP


# =============================================================================
# 여러 가지 입력을 BGR 배열로 통일
#
# 아래 코드는 딱 한 가지 형태만 받습니다. 연속된 BGR uint8 배열로, OpenCV와 InsightFace가
# 모두 요구하는 형식입니다.
# =============================================================================

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def to_bgr_array(image: ImageInput, input_is_bgr: bool = True) -> np.ndarray:
    """경로, 바이트, 배열을 모두 연속된 BGR uint8 배열로 통일합니다.

    input_is_bgr=True는 들어온 배열이 이미 BGR 순서라는 뜻입니다（OpenCV 자체가 이
    순서이며 기본값입니다）. False를 넘기면 RGB이므로 여기서 BGR로 뒤집습니다. 경로나
    바이트를 넘길 때는 이 인자가 영향을 주지 않습니다.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            array = np.repeat(image[:, :, None], 3, axis=2)      # 그레이스케일을 3채널로
        elif input_is_bgr:
            array = image
        else:
            array = image[:, :, ::-1]                              # RGB를 BGR로
        if array.dtype != np.uint8:
            # 또 다른 흔한 규약은 [0, 1] 범위의 실수 배열입니다.
            if np.issubdtype(array.dtype, np.floating) and array.size > 0:
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype("uint8")
        return np.ascontiguousarray(array)

    # cv2.imread는 Windows의 한자 경로를 읽지 못하므로 바이트를 직접 읽어 디코딩합니다.
    # 모든 입력 종류가 이 경로를 탑니다.
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지가 없습니다: {image_path}")
        data = image_path.read_bytes()

    decoded = cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("이미지를 디코딩할 수 없습니다. 손상되었거나 이미지가 아닐 수 있습니다")
    return np.ascontiguousarray(decoded)


# =============================================================================
# 특징 추출, 점수 계산
#
# 이 부분이 엔진 전체입니다. 웹 쪽은 이미지를 옮기고 JSON을 옮길 뿐, 다른 일은 하지 않습니다.
# =============================================================================


def _face_box_area(face) -> float:
    """검출 상자의 면적. 단위는 픽셀입니다."""
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
    """특징 벡터를 자신의 길이로 나눕니다. 전부 0인 벡터면 None을 반환합니다."""
    array = np.asarray(vector, dtype="float32").reshape(-1)
    length = float(np.linalg.norm(array))
    if length == 0.0:
        return None
    return array / length


def encode_faces(image: ImageInput, input_is_bgr: bool = True) -> list:
    """이미지의 얼굴마다 512차원 특징 하나. 이미 L2 정규화되어 있습니다.

    얼굴 상자 면적이 큰 순서로 정렬되어, 0번이 그 사진에서 가장 큰 얼굴입니다. 호출자는
    이미지에 얼굴이 몇 장 있는지를 더 따질 필요가 없습니다. 얼굴을 못 찾으면 빈 목록을 반환합니다.
    """
    detected = encode_faces_with_scores(image, input_is_bgr)
    embeddings = []
    for face in detected:
        embeddings.append(face["embedding"])
    return embeddings


def encode_faces_with_scores(
    image: ImageInput, input_is_bgr: bool = True
) -> list:
    """encode_faces와 같지만, 검출기의 신뢰도와 상자 면적을 함께 돌려줍니다.

    각 항목은 {"embedding": 특징, "det_score": 검출 신뢰도, "area": 상자 면적}입니다.
    검출기가 얼마나 확신하는지 알고 싶을 때만 쓰세요.
    """
    faces = get_app().get(to_bgr_array(image, input_is_bgr))

    results = []
    for face in faces:
        # 여기에 `or []`라고 쓸 수 없습니다. 특징은 NumPy 배열이라 배열의 참·거짓 판정이
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

    # 큰 것이 앞에 오므로 호출자가 0번을 곧바로 "이 사진의 사람"으로 쓸 수 있습니다.
    results.sort(key=lambda row: row["area"], reverse=True)
    return results


def cosine_similarity_percent(first: np.ndarray, second: np.ndarray) -> float:
    """두 특징의 코사인 유사도를 퍼센트로 환산합니다."""
    left = np.asarray(first, dtype="float32").reshape(-1)
    right = np.asarray(second, dtype="float32").reshape(-1)
    if left.size != right.size:
        raise ValueError(f"특징 길이가 서로 다릅니다: {left.size} 와 {right.size}")
    return float(np.dot(left, right)) * 100.0


def best_pair_percent(first_faces: Sequence, second_faces: Sequence,
                      largest_only: bool = False) -> float:
    """두 사진 사이에서 가장 높은 점수.

    그룹 사진은 기본 설정만으로 연결됩니다. 양쪽에 얼굴이 몇 장 있는지는 신경 쓰지
    않습니다. largest_only=True면 양쪽에서 가장 큰 얼굴끼리만 비교합니다.
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
    """후보마다 최고 점수를 하나씩 내고, 후보의 원래 순서대로 정렬합니다.

    질의 이미지에 얼굴이 여러 장일 수 있으므로（그룹 사진）, 각 후보는 질의 이미지에서
    가장 잘 맞는 얼굴과 비교합니다. Python에서 중첩 반복으로 돌리는 대신 행렬 곱셈 한 번으로
    끝내므로, 얼굴 라이브러리가 커질 때 차이가 뚜렷합니다.
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
    """두 이미지 사이의 얼굴 유사도. 퍼센트입니다.

    input_is_bgr의 뜻은 to_bgr_array를 보세요. largest_only=True면 각 이미지에서 가장 큰
    얼굴 하나만 골라 비교하고, 기본값은 모든 얼굴을 서로 짝지어 비교해 최고점을 취하므로

    그룹 사진도 얼굴을 먼저 잘라낼 필요가 없습니다. 한쪽 이미지에서 얼굴을 못 찾으면 None을
    반환합니다. 점수가 높을수록 닮았다는 뜻이며, 실측 기준 같은 사람의 서로 다른 사진은
    """
    faces_a = encode_faces(image_a, input_is_bgr)
    faces_b = encode_faces(image_b, input_is_bgr)
    if not faces_a or not faces_b:
        return None

    score = best_pair_percent(faces_a, faces_b, largest_only)
    # 부동소수점 오차로 결과가 범위를 조금 벗어날 수 있어, -100~100으로 눌러 둡니다.
    if score > 100.0:
        return 100.0
    if score < -100.0:
        return -100.0
    return score


# =============================================================================
# 얼굴 라이브러리
#
# 얼굴 라이브러리는 이미지가 들어 있는 어떤 폴더로나 가능합니다. 전체를 계산하려면 시간이
# 꽤 걸리므로 파일마다 한 번만 계산해 캐시하고, 파일의 수정 시각으로 변동 여부를 봅니다.
# 수정·추가·삭제가 있으면 다음 비교에서 드러납니다.
# =============================================================================


def library_images() -> list:
    """얼굴 라이브러리 폴더의 모든 이미지. 하위 폴더 포함, 순서는 고정입니다."""
    found = []
    if not LIBRARY_FOLDER.is_dir():
        return found
    for path in LIBRARY_FOLDER.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    found.sort()
    return found


def _embeddings_of(path: Path) -> list:
    """한 파일에 있는 모든 얼굴의 특징. 파일이 변하지 않았다면 캐시를 그대로 씁니다.

    읽을 수 없거나 얼굴이 없으면 오류를 내지 않고 빈 목록을 반환합니다. 이미지 하나가
    안 되다고 얼굴 라이브러리 전체가 무너져서는 안 됩니다.
    """
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        return []          # 스캔 직후에 삭제됨

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


def _load_library_full() -> tuple:
    """얼굴 라이브러리의 모든 얼굴, (이름 목록, 특징 행렬, 상대 경로 목록)을 반환합니다.

    상대 경로는 각 얼굴이 어느 사진에서 왔는지 라이브러리 안에서의 위치입니다. 한 줄이
    이름 하나에 대응하며 순서도 같습니다. 결과 행이 자신이 적은 사진으로 되돌아갈 수 있게 하려고 둘 뿐이며, 절대 경로는 결코 외부로 나가지 않습니다.
    """
    images = library_images()
    labels = []
    embeddings = []
    relative_paths = []
    alive = []

    # 한 번에 한 명만 라이브러리를 훑게 합니다. ONNX 세션과 검출 네트워크는 두 요청이
    with LIBRARY_LOCK:
        for index, path in enumerate(images, start=1):
            if len(images) >= 20:
                print(f"  얼굴 라이브러리 로드 중 {index}/{len(images)}  {path.name}", flush=True)

            alive.append(str(path))
            relative_path = path.relative_to(LIBRARY_FOLDER).as_posix()
            position = 0
            for embedding in _embeddings_of(path):
                position += 1
                if position == 1:
                    labels.append(path.name)
                else:
                    labels.append(f"{path.name} #{position}")
                embeddings.append(embedding)
                relative_paths.append(relative_path)

        # 얼굴 라이브러리에서 삭제된 이미지의 캐시도 함께 지웁니다.
        for key in list(LIBRARY_CACHE):
            if key not in alive:
                del LIBRARY_CACHE[key]

    if not embeddings:
        return labels, np.zeros((0, 512), dtype="float32"), relative_paths
    return labels, np.vstack(embeddings).astype("float32"), relative_paths


def load_library() -> tuple:
    """얼굴 라이브러리의 모든 얼굴, (이름 목록, 특징 행렬)을 반환합니다.

    이름은 곧 파일명이며, 사진 한 장에 얼굴이 여러 장 있으면 뒤에 " #2", " #3"을 붙입니다.
    특징 행렬은 이름 하나에 한 줄이 대응하며 순서도 같습니다.
    """
    labels, matrix, _relative_paths = _load_library_full()
    return labels, matrix


def library_image_url(relative_path: str) -> str:
    """얼굴 라이브러리 사진 한 장을 브라우저가 쓸 주소.

    경로는 이스케이프합니다. 하위 폴더 이름에 공백이나 한자가 들어 있을 수 있는데, 이는
    폴더 이름으로는 합법이고 URL에서도 합법이지만 먼저 이스케이프해야 하기 때문입니다.
    """
    return f"{LIBRARY_IMAGE_ROUTE}/{quote(relative_path)}"


# =============================================================================
# 점수를 결과 표로 바꾸기
# =============================================================================


def rank_matches(
    query_faces: list,
    labels: list,
    candidate_faces: Sequence,
    group_label: str,
    top_results: int,
    image_paths: Optional[Sequence] = None,
) -> list:
    """각 후보를 질의 이미지와 한 번씩 비교하고, 상위 몇 개만 남깁니다.

    labels는 각 후보를 표시할 이름이며 candidate_faces와 순서가 같습니다. group_label은
    후보가 어디에서 왔는지를 나타내며, 각 행의 "kind"로 표시됩니다（업로드한 것은 "후보 이미지",
    얼굴 라이브러리에 있는 것은 "라이브러리 후보"）.

    반환되는 행은 {"rank", "kind", "candidate", "similarity"}이며, 1위에는
    "best": True가 하나 더 붙습니다. 후보가 없으면 빈 목록을 반환합니다.

    image_paths를 넘기면 각 이름에 대응하는 라이브러리 내 사진이 순서대로 들어와, 각 행에
    썸네일 주소가 하나씩 붙습니다. 넘기지 않은 행（예컨대 이번에 업로드한 후보 이미지）은 애초에 보여줄 이미지가 없습니다.
    """
    scores = best_scores_percent(query_faces, candidate_faces)
    order = np.argsort(-scores)
    count = min(len(scores), max(1, int(top_results)))

    matches = []
    for position in range(count):
        index = int(order[position])
        row = {
            "rank": position + 1,
            "kind": group_label,
            "candidate": labels[index],
            "similarity": round(float(scores[index]), 2),
        }
        if image_paths is not None:
            row["image_url"] = library_image_url(image_paths[index])
        matches.append(row)
    if matches:
        matches[0]["best"] = True
    return matches


def disambiguate(names: list) -> list:
    """이름이 겹치는 파일 뒤에 번호를 붙여, 결과 표의 두 줄이 똑같아지지 않게 합니다."""
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
# 웹
# =============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


class RequestError(Exception):
    """이번 요청에 문제가 있습니다. 메시지는 사용자에게 보입니다."""


def error_response(message: str, status: int = 400):
    """페이지와 API를 직접 호출하는 사람이 똑같이 보는 오류 형식입니다."""
    return jsonify({"ok": False, "error": message}), status


@app.errorhandler(RequestError)
def handle_request_error(error: RequestError):
    return error_response(str(error))


@app.errorhandler(413)
def handle_too_large(_error):
    return error_response("업로드 파일이 너무 큽니다. 요청당 최대 64 MB", 413)


def read_uploads(field_name: str) -> list:
    """어떤 필드에 업로드된 모든 파일, (파일 이름, 바이트 내용)."""
    uploads = []
    for item in request.files.getlist(field_name):
        if item and item.filename:
            uploads.append((item.filename, item.read()))
    return uploads


def encode_upload(data: bytes, label: str, file_name: str) -> list:
    """업로드한 이미지 한 장에 있는 모든 얼굴의 특징.

    이미지를 쓸 수 없으면（디코딩 실패, 손상, 얼굴 없음）RequestError를 바로 내고, 메시지에
    어느 이미지인지 함께 넣습니다. label은 "질의 이미지" 또는 "후보 이미지"입니다.
    """
    try:
        faces = encode_faces(data)
    except (OSError, ValueError) as error:
        raise RequestError(f"{label}처리 실패: {error}: {file_name}")
    if not faces:
        raise RequestError(f"{label}얼굴이 검출되지 않았습니다: {file_name}")
    return faces


def query_from_request() -> tuple:
    """이번 요청의 질의 이미지, (파일 이름, 모든 얼굴의 특징)."""
    uploads = read_uploads("query")
    if not uploads:
        raise RequestError("질의 이미지 한 장을 선택해 주세요")
    if len(uploads) > 1:
        raise RequestError("질의 이미지는 한 장만 선택할 수 있습니다")

    name, data = uploads[0]
    return name, encode_upload(data, "질의 이미지", name)


def top_results_from_request() -> int:
    """앞으로 몇 개를 반환할지. 범위를 벗어난 값은 경계값으로 계산합니다."""
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


# -------------------- 페이지 --------------------
PAGE = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>얼굴 유사도</title>
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
  .thumb {
    width: 2.75rem; height: 2.75rem; object-fit: cover;
    border-radius: .35rem; border: 1px solid #f1e2d6; background: #faf5f1;
    display: block;
  }
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
      <h1 class="h3 mb-1" style="color:#7a2e0a">얼굴 유사도</h1>
      <div class="text-secondary">업로드 비교 · 얼굴 라이브러리 검색</div>
    </div>
    <div class="text-end">
      <span class="badge rounded-pill text-bg-light border" id="libraryBadge">얼굴 라이브러리 로드 중…</span>
    </div>
  </div>

  <div class="row g-3 align-items-stretch">
    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">1</span>질의 이미지 vs 후보 이미지（1:N）</div>
          <p class="form-text mb-3">질의 이미지 한 장으로 아래에서 고른 후보 이미지를 하나씩 비교하며, 로컬 얼굴 라이브러리는 <strong>사용하지 않습니다</strong>.</p>
          <form id="queryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="queryFile">① 질의 이미지（한 장）</label>
              <input class="form-control form-control-lg" type="file" id="queryFile"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="queryFile">예: 누군지 검색.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="queryFile"></div>
            </div>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="candidateFiles">② 후보 이미지（여러 장 가능）</label>
              <input class="form-control form-control-lg" type="file" id="candidateFiles"
                     name="candidates" accept="image/*" multiple required>
              <div class="form-text" data-summary="candidateFiles">예: 사용자1.jpg、사용자2.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="candidateFiles"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="queryTop">반환 개수</label>
              <input class="form-control form-control-lg" type="number" id="queryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="querySubmit">
              비교 시작
            </button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">2</span>질의 이미지 vs 얼굴 라이브러리（1:N）</div>
          <p class="form-text mb-3">질의 이미지 한 장으로 얼굴 라이브러리의<b>모든</b>이미지를 비교해, 가장 비슷한 몇 장을 자동으로 돌려줍니다.</p>
          <form id="libraryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="libraryQuery">① 질의 이미지（한 장）</label>
              <input class="form-control form-control-lg" type="file" id="libraryQuery"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="libraryQuery">예: 누군지 검색.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="libraryQuery"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="libraryTop">반환 개수</label>
              <input class="form-control form-control-lg" type="number" id="libraryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="librarySubmit">
              얼굴 라이브러리 검색
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
            <div class="panel-title mb-0"><span class="step">3</span>결과</div>
            <div class="form-text" id="resultMeta"></div>
          </div>
          <div id="resultAlert"></div>
          <div id="resultBody" class="empty-state">아직 비교를 시작하지 않았습니다.</div>
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
    ? '<span class="spinner me-2"></span>비교 중…'
    : idleText;
}

function showError(message) {
  $("resultAlert").innerHTML =
    '<div class="alert alert-brand alert-dismissible fade show" role="alert">' +
    '<strong>오류:</strong>' + escapeHtml(message) +
    '<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>';
  $("resultBody").innerHTML = "";
  $("resultMeta").textContent = "";
}

function resultTable(rows) {
  if (!rows || !rows.length) {
    return '<p class="empty-state mb-0">비교할 대상이 없습니다.</p>';
  }
  // The bar length is the similarity itself on a fixed 0-100 scale, so it
  // always matches the percentage printed next to it.
  const body = rows.map((row) => {
    const span = Math.max(0, Math.min(100, row.similarity));
    // image_url이 없는 행은 이번에 업로드한 후보 이미지이지 라이브러리 속 사진이 아니므로, 보여줄 그림이 없습니다.
    const thumb = row.image_url
      ? `<img class="thumb" src="${encodeURI(row.image_url)}" alt="" loading="lazy">`
      : "";
    return `
    <tr${row.best ? ' class="table-warning"' : ''}>
      <td style="width:4rem">${row.rank}</td>
      <td style="width:3.5rem">${thumb}</td>
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
    <thead><tr><th>순위</th><th></th><th>${escapeHtml(rows[0].kind || "후보")}</th><th>유사도</th><th></th></tr></thead>
    <tbody>${body}</tbody></table>
    <p class="form-text mt-2 mb-0">가로 막대 길이가 그대로 유사도 퍼센트이며, 가득 차면 100%입니다.</p>`;
}

function skippedList(skipped) {
  if (!skipped || !skipped.length) return "";
  return '<p class="form-text mt-3 mb-0">건너뜀:'
    + skipped.map((item) => escapeHtml(item.name) + "（" + escapeHtml(item.status) + "）").join("、")
    + '</p>';
}

function verdictBlock(matches) {
  if (!matches || !matches.length) {
    return '<p class="empty-state">사용할 수 있는 후보가 없습니다.</p>';
  }
  const best = matches[0];
  return `<div class="verdict d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div>
        <div class="form-text mb-1">가장 비슷한 항목</div>
        <div class="who">${escapeHtml(best.candidate)}</div>
      </div>
      <div class="text-end">
        <div class="form-text mb-1">유사도</div>
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
  if (payload.query) meta.push("질의: " + payload.query);
  if (payload.elapsed_ms != null) meta.push("총 소요 시간 " + payload.elapsed_ms + " ms");
  if (payload.count != null) meta.push("처리 " + payload.count + " 개");
  if (payload.library_faces != null) meta.push("얼굴 라이브러리 " + payload.library_faces + "개의 얼굴");
  if (payload.library_images != null) meta.push("얼굴 라이브러리 " + payload.library_images + "장의 이미지");
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
      throw new Error("서버에서 해석할 수 없는 내용을 반환했습니다（HTTP " + response.status + "）");
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || ("요청 실패（HTTP " + response.status + "）"));
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
        showError("이 패널에 필요한 이미지를 먼저 선택해 주세요");
        return;
      }
    }
    postJson(url, new FormData(form), $(buttonId), idleText);
  });
}

bindForm("queryForm", "querySubmit", "/api/query-set", "비교 시작",
         ["queryFile", "candidateFiles"]);
bindForm("libraryForm", "librarySubmit", "/api/query-library", "얼굴 라이브러리 검색",
         ["libraryQuery"]);

document.querySelectorAll('input[type="file"]').forEach((input) => {
  input.addEventListener("change", () => {
    const picked = Array.from(input.files);
    const summary = document.querySelector('[data-summary="' + input.id + '"]');
    if (summary && input.multiple) {
      summary.textContent = picked.length
        ? "이미지 " + picked.length + "장 선택됨"
        : "아직 선택되지 않음";
    } else if (summary && picked.length) {
      summary.textContent = "선택됨: " + picked[0].name;
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
        ? "얼굴 라이브러리 " + payload.faces + "개의 얼굴 / " + payload.images + "장의 이미지"
        : "얼굴 라이브러리를 사용할 수 없음";
    })
    .catch(() => { $("libraryBadge").textContent = "얼굴 라이브러리 상태 알 수 없음"; });
}

loadLibraryStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """페이지 자체. 두 패널이 모두 여기에 있습니다. 이 페이지에는 검출 방식 드롭다운이 없습니다.

    숫자 세 개는 페이지의 "앞으로 몇 개를 표시할지" 입력창의 초깃값과 상·하한이며,
    top_results_from_request의 경계와 맞춰져 있습니다.
    """
    return render_template_string(PAGE, top_results=10, top_results_min=1, top_results_max=100)


@app.get("/api/library")
def api_library():
    """얼굴 라이브러리에 이미지가 몇 장, 얼굴이 몇 개 있는지.

    숫자만 돌려주며, 폴더 경로 자체는 절대로 서버 밖으로 나가지 않습니다.
    """
    _labels, embedding_matrix = load_library()
    return jsonify(
        {
            "ok": True,
            "images": len(library_images()),
            "faces": int(embedding_matrix.shape[0]),
        }
    )


@app.get("/api/health")
def api_health():
    """스크립트와 프로세스 관리자가 쓰는 "준비됐나요".

    일부러 아주 가볍게 만들었습니다. 얼굴 라이브러리의 이미지만 세고 특징 추출은 전혀 하지
    않으므로, 프로브가 몇 초마다 물어도 얼굴 검출 비용을 소모하지 않습니다. "ready"는
    폴더에 이미지가 한 장 이상 있다는 뜻이며, 그 이미지들에 실제로 얼굴이 있는지는 실제로 검색할 때야 드러납니다.
    """
    images = library_images()
    return jsonify(
        {
            "ok": True,
            "ready": bool(images),
            "images": len(images),
            "descriptor_dimension": 512,
        }
    )


@app.get("/api/stats")
def api_stats():
    """이 컴퓨터에서 서비스가 지금 실제로 쓰는 설정.

    점수를 결정하는 설정들（모델, 특징 차원, 임계값 방향, 라이브러리 규모）과 버전 번호입니다.
    두 컴퓨터의 결과가 달라서 이유를 알고 싶을 때면 이것만 보면 됩니다. 얼굴 라이브러리
    폴더는 이름만 알리고 경로는 알려주지 않습니다.
    """
    directory = find_model_directory()
    _labels, matrix, _image_paths = _load_library_full()

    missing = (
        list(REQUIRED_MODEL_FILES)
        if directory is None
        else [name for name in REQUIRED_MODEL_FILES if not (directory / name).is_file()]
    )

    return jsonify(
        {
            "ok": True,
            "library_images": len(library_images()),
            "library_faces": int(matrix.shape[0]),
            "descriptor_dimension": 512,
            "threshold_metric": "cosine_similarity",
            "sort_order": "cosine_similarity_descending",
            "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            "onnxruntime_version": onnxruntime.__version__,
            "onnxruntime_providers": onnxruntime.get_available_providers(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "model_directory_present": directory is not None,
            "missing_models": missing,
            "model_summary": MODEL_SUMMARY,
        }
    )


@app.get(LIBRARY_IMAGE_ROUTE + "/<path:relative_path>")
def library_image(relative_path: str):
    """얼굴 라이브러리의 사진 한 장. 결과 표의 썸네일에 쓰입니다.

    send_from_directory는 경로를 얼굴 라이브러리 폴더 안에 붙잡고, 바깥으로 나가려는
    어떤 기법도 거부하므로, 경로를 어떻게 조합해도 이 컴퓨터의 다른 파일에는 닿지 않습니다.
    """
    return send_from_directory(LIBRARY_FOLDER, relative_path)


@app.get(LIBRARY_IMAGE_DOWNLOAD_ROUTE + "/<path:relative_path>")
def library_image_download(relative_path: str):
    """같은 사진을 내려받는 형태. 어떤 줄을 저장하거나 따로 열어 볼 때 편합니다."""
    return send_from_directory(LIBRARY_FOLDER, relative_path, as_attachment=True)


@app.post("/api/query-set")
def api_query_set():
    """패널 하나: 질의 이미지를 이번에 함께 업로드한 후보 이미지와 비교합니다."""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    uploads = read_uploads("candidates")
    if not uploads:
        raise RequestError("후보 이미지를 한 장 이상 선택해 주세요")

    # 쓸 수 없는 이미지가 있으면 적어 두고 건너뛰며, 전체 비교에는 영향이 없습니다.
    labels = []
    candidate_faces = []
    skipped = []
    for name, data in uploads:
        try:
            faces = encode_upload(data, "후보 이미지", name)
        except RequestError as error:
            skipped.append({"name": name, "status": str(error)})
            continue
        labels.append(name)
        candidate_faces.append(faces[0])        # 이 사진에서 가장 큰 얼굴

    if not candidate_faces:
        raise RequestError("후보 이미지에서 사용할 수 있는 얼굴이 없습니다")

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
                "후보 이미지",
                top_results_from_request(),
            ),
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.post("/api/query-library")
def api_query_library():
    """패널 둘: 질의 이미지를 얼굴 라이브러리의 모든 이미지와 비교합니다."""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    if not library_images():
        raise RequestError("얼굴 라이브러리에 이미지가 없습니다. 먼저 이미지를 넣고 다시 시도해 주세요")

    labels, embedding_matrix, image_paths = _load_library_full()
    if embedding_matrix.shape[0] == 0:
        raise RequestError("얼굴 라이브러리에서 얼굴이 검출되지 않았습니다. 먼저 얼굴이 있는 이미지를 넣고 다시 시도해 주세요")

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
                "라이브러리 후보",
                top_results_from_request(),
                image_paths,
            ),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


# =============================================================================
# 시작
# =============================================================================


def ask_library_folder() -> Path:
    """시작할 때 얼굴 라이브러리가 어디 있는지 한 번 물어보고, 그냥 Enter를 누르면 스크립트와 같은 위치의 face_library를 씁니다.

    물어볼 창이 없을 때는（예컨대 더블클릭으로 실행했거나 다른 프로그램이 띄운 경우）묻지 않고 기본 위치를
    그대로 쓰며, 이미 닫힌 입력 스트림을 읽지 않습니다.
    """
    default = SCRIPT_DIRECTORY / "face_library"
    if not sys.stdin.isatty():
        default.mkdir(parents=True, exist_ok=True)
        return default

    print("얼굴 라이브러리는 이미지를 넣어 두는 폴더입니다. 하위 폴더가 있어도 됩니다.")
    print(f"지원 형식: {' '.join(sorted(SUPPORTED_EXTENSIONS))}")
    while True:
        answer = input(f"얼굴 라이브러리 폴더（그냥 Enter면 {default}）: ").strip()
        if not answer:
            default.mkdir(parents=True, exist_ok=True)
            return default
        folder = Path(answer).expanduser()
        if folder.is_dir():
            return folder
        if folder.exists():
            print("파일이며 폴더가 아닙니다. 다시 입력해 주세요.")
        else:
            print("이 폴더는 존재하지 않습니다. 경로를 확인한 뒤 다시 입력해 주세요.")


def choose_port(preferred: int = 5000) -> int:
    """preferred부터 뒤로 넘겨 보면서 아무도 쓰지 않는 포트를 찾습니다.

    5000번 포트는 자주 다른 프로그램이 먼저 잡고 있어서, 잡혀 있으면 자동으로 뒤로 미뤄 시작 실패를 피합니다.
    """
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # 여기서는 SO_REUSEADDR를 설정하지 않습니다. Windows에서 설정하면 이 탐색이 다른
            # 프로그램이 듣는 중인 포트를 물 수 있습니다.
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"{preferred} 부터 {preferred + 19} 사이에는 빈 포트가 없습니다")


def main():
    """얼굴 라이브러리가 어디 있는지 물어보고, 모델을 로드한 다음 웹 서비스를 띄웁니다."""
    global LIBRARY_FOLDER
    LIBRARY_FOLDER = ask_library_folder()

    print("buffalo_l 모델을 불러오는 중입니다. 약 2초…", flush=True)
    load_model()
    print("모델 로드 완료")
    print(MODEL_SUMMARY, flush=True)
    print(f"얼굴 라이브러리 경로: {LIBRARY_FOLDER}", flush=True)

    images = library_images()
    print(f"얼굴 라이브러리에 현재 {len(images)}장의 이미지가 있습니다", flush=True)

    port = choose_port()
    print(f"Open http://127.0.0.1:{port} in your browser", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()