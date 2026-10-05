"""顔の類似度 Web 版（InsightFace）。

起動すると、まず顔画像がどのフォルダにあるかを尋ね、答えが返れば Web サービスを立ち上げる：ページを開き、
クエリ画像1枚と候補画像の一式を選ぶと、各候補画像がクエリ画像とどれくらい似ているかが分かる。あるいは
クエリ画像を自分の顔フォルダと比較。顔は SCRFD で検出し、ArcFace で記述し、各顔は
512次元の特徴（L2 正規化済み）になり、スコアは2つの顔の特徴のコサイン類似度に100を掛けた値。同一人物の
異なる写真での実測値は 77-79、他人なら -3 ~ +1。同一人物かどうかは判定せず、
閾値は自分のサンプルで別途決める必要がある。

スコアは ArcFace 由来で、dlib 版（flask_face_match_v2.py）とは別のモデルなので数値は
互いに比較できない。Web ページのレイアウト、配色、そしてあの固定の 0-100 の類似度バーは dlib 版と同じで、
違うのは2点だけ：検出方式のドロップダウンがない（SCRFD が唯一の検出器）、結果にも
detector フィールドがない。

依存関係のインストール（Python 3.9 以上）：

    pip install flask opencv-python numpy insightface onnxruntime

NVIDIA の GPU で使うなら、onnxruntime を onnxruntime-gpu に置き換える。

起動：

    python flask_insightface_face_v3.py

そして指示に従って顔ライブラリフォルダを入力する。何も入力せず Enter を押すと、スクリプトと同じ階層の face_library を使う。画像形式は
.jpg .jpeg .png .bmp .webp。サブディレクトリも置ける。ページに表示することも、パスを受け取ることもないので、
この場所は起動した本人だけが知っている。

buffalo_l モデル（約 300 MB）は初回実行時に ~/.insightface/models/
buffalo_l へ自動ダウンロードされる。スクリプトと同じ階層に insightface_models/buffalo_l か models/buffalo_l が既にある場合は
その場で使用、オンラインにはしない。

設定はすべてコード内に書かれていて、実行時に必要なものだけ尋ねる。環境変数は一切読まない。
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

# buffalo_l のうち読み込むのは検出と認識の2モデルだけ。キーポイントや性別・年齢のモデルは不要、
# それらを読み込むと起動が遅くなるだけ。
ALLOWED_MODULES = ["detection", "recognition"]
REQUIRED_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
MODEL_FOLDER = Path("~/.insightface").expanduser() / "models" / "buffalo_l"
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# InsightFace のキーポイントモデルが呼ぶ scikit-image の旧インターフェース。顔を1枚検出するたびに警告が
# 1回出る。類似度の計算にはこの警告は不要なので、ここでまとめて1回だけ切る。
warnings.filterwarnings("ignore", message=r".*estimate.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.*")

# 顔ライブラリのフォルダ。起動時の質問で決まる。以後ずっとこの値を使う。
LIBRARY_FOLDER = SCRIPT_DIRECTORY / "face_library"

# 結果表の各行がサムネイルを1枚表示するため、ブラウザがこれらの画像をもう一度取得できる必要がある。公開するのは
# ルートだけで、フォルダは公開しない：リクエストに書けるのは顔ライブラリ内の相対パスだけ。
LIBRARY_IMAGE_ROUTE = "/library-image"
LIBRARY_IMAGE_DOWNLOAD_ROUTE = "/library-image-download"

# ロード済みの FaceAnalysis。構築に2秒超かかるので一度だけ作り、以降の各リクエストはこれを使う。
FACE_APP = None
MODEL_SUMMARY = ""

# フォルダ -> (更新時刻, そのファイルの各顔の特徴)
LIBRARY_CACHE = {}
LIBRARY_LOCK = threading.Lock()


# =============================================================================
# モデルの探索と読み込み
# =============================================================================


def find_model_directory() -> Optional[Path]:
    """ディスク上にすでに 있는 buffalo_l ディレクトリ。見つからなければ None を返す。

    スクリプトと同じ階層を優先するので、モデルをプログラムの隣に置けばオフラインで実行できる。本当に無いときだけ
    InsightFace にダウンロードさせる。
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
    """検出・認識モデルを読み込み、ログ用の説明を1行で返す。

    このプロセス内で一度だけ実行し、以降のリクエストは読み込んだ FACE_APP をそのまま使う。
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
        # MODEL_FOLDER へのダウンロードは InsightFace 自身に任せる。
        FACE_APP = FaceAnalysis(
            name="buffalo_l",
            root=str(MODEL_FOLDER.parents[1]),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = f"{MODEL_FOLDER}（今回のダウンロード）"
    else:
        FACE_APP = FaceAnalysis(
            name=str(found),
            providers=providers,
            allowed_modules=ALLOWED_MODULES,
        )
        where = str(found)

    # ctx_id は CUDA バックエンドのときだけ意味を持つ。
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    FACE_APP.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.5)

    MODEL_SUMMARY = (
        f"model=buffalo_l dim=512 providers={','.join(providers)} models={where}"
    )
    return MODEL_SUMMARY


def get_app():
    """ロード済みのモデルを返す。必要なら先に読み込む。"""
    if FACE_APP is None:
        load_model()
    return FACE_APP


# =============================================================================
# 各種入力を BGR 配列に統一
#
# 以下のコードは1つの形状しか認識しない：連続した BGR uint8 配列。OpenCV と InsightFace は
# どちらもこの形式を要求する。
# =============================================================================

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def to_bgr_array(image: ImageInput, input_is_bgr: bool = True) -> np.ndarray:
    """パス・バイト列・配列を、連続した BGR uint8 配列に統一する。

    input_is_bgr=True は渡された配列が既に BGR 順であることを示す（OpenCV 自体もこの
    順序がデフォルト値）；False なら既に RGB という意味で、ここでは BGR に反転する。パスやバイト列を渡すとき
    このパラメータは無視される。
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            array = np.repeat(image[:, :, None], 3, axis=2)      # グレースケールを3チャンネルに展開
        elif input_is_bgr:
            array = image
        else:
            array = image[:, :, ::-1]                              # RGB を BGR に変換
        if array.dtype != np.uint8:
            # もう1つよくある約束として [0, 1] の浮動小数点配列がある。
            if np.issubdtype(array.dtype, np.floating) and array.size > 0:
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype("uint8")
        return np.ascontiguousarray(array)

    # cv2.imread は Windows 上の中国語パスを読めないので、自分でバイト列を読んでデコードする。
    # すべての入力種別がこの経路を通る。
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"画像が存在しません：{image_path}")
        data = image_path.read_bytes()

    decoded = cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("画像をデコードできません。破損しているか画像ではありません")
    return np.ascontiguousarray(decoded)


# =============================================================================
# 特徴の抽出とスコアの計算
#
# この部分がエンジン本体。Web 側は画像と JSON を運ぶだけで、ほかは何もしない。
# =============================================================================


def _face_box_area(face) -> float:
    """検出ボックスの面積、単位はピクセル。"""
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
    """特徴ベクトルを自身の長さで割る。ゼロベクトルなら返す None。"""
    array = np.asarray(vector, dtype="float32").reshape(-1)
    length = float(np.linalg.norm(array))
    if length == 0.0:
        return None
    return array / length


def encode_faces(image: ImageInput, input_is_bgr: bool = True) -> list:
    """画像内の各顔につき1つ、L2 正規化済みの512次元特徴を返す。

    顔ボックスの面積の降順に並ぶので、0番目が写真で最も大きい顔になる。呼び出し側は
    1枚に何枚の顔があるか気にする必要はない。顔が検出されなければ空リストを返す。
    """
    detected = encode_faces_with_scores(image, input_is_bgr)
    embeddings = []
    for face in detected:
        embeddings.append(face["embedding"])
    return embeddings


def encode_faces_with_scores(
    image: ImageInput, input_is_bgr: bool = True
) -> list:
    """encode_faces と同様に、検出器自身の信頼度とボックス面積も付く。

    各項目は {"embedding": 特徴, "det_score": 検出信頼度, "area": ボックス面積}。
    検出器がどれだけ確信しているかを調べたいときだけ必要。
    """
    faces = get_app().get(to_bgr_array(image, input_is_bgr))

    results = []
    for face in faces:
        # ここには `or []` と書けない：特徴は NumPy 配列で、配列の真偽判定はエラーになる
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

    # 面積の大きい順なので、呼び出し側は 0 番目をそのまま"この写真の人"として使える。
    results.sort(key=lambda row: row["area"], reverse=True)
    return results


def cosine_similarity_percent(first: np.ndarray, second: np.ndarray) -> float:
    """2つの特徴のコサイン類似度を、百分率に換算して返す。"""
    left = np.asarray(first, dtype="float32").reshape(-1)
    right = np.asarray(second, dtype="float32").reshape(-1)
    if left.size != right.size:
        raise ValueError(f"特徴の長さが不一致：{left.size} と {right.size}")
    return float(np.dot(left, right)) * 100.0


def best_pair_percent(first_faces: Sequence, second_faces: Sequence,
                      largest_only: bool = False) -> float:
    """2枚の写真間の最高スコア。

    集合写真でもデフォルトでマッチングし、各辺に何枚の顔があるか気にする必要はない。largest_only=True のときは
    各辺の最大の顔だけを比較する。
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
    """各候補の最高スコアを1つずつ、候補の元の順に並べて返す。

    クエリ画像には複数の顔（集合写真）があるので、各候補はクエリ画像で最も合う顔と比較する。
    行列積1回で計算するので Python での二重ループは使わない。顔ライブラリが大きいほど差が
    はっきりする。
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
    """2枚の画像間の顔の類似度を、百分率で返す。

    input_is_bgr の意味は to_bgr_array を参照。largest_only=True のときは各画像の最大
    の顔だけを比較する。デフォルトでは全ての顔を総当たりで比較して最高点を取る。集合写真でも事前に顔は切り出さない。

    どちらかの画像で顔が検出されなければ None を返す。スコアが高いほど似ている。実測では同一人物の異なる写真で
    77-79、他人なら -3 ~ +1。
    """
    faces_a = encode_faces(image_a, input_is_bgr)
    faces_b = encode_faces(image_b, input_is_bgr)
    if not faces_a or not faces_b:
        return None

    score = best_pair_percent(faces_a, faces_b, largest_only)
    # 浮動小数点の誤差で結果がわずかに範囲を超えることがあるので、-100 ~ 100 に押し戻す。
    if score > 100.0:
        return 100.0
    if score < -100.0:
        return -100.0
    return score


# =============================================================================
# 顔ライブラリ
#
# 顔ライブラリは画像を置いた任意のフォルダ。全体を走査するには時間がかかるので、各ファイルについて
# 1回だけ計算してキャッシュし、ファイルの更新時刻で変化を判断する：変更・追加・削除があれば、
# 次回の比較でわかる。
# =============================================================================


def library_images() -> list:
    """顔ライブラリフォルダ内のすべての画像。サブディレクトリも含む。順序は固定。"""
    found = []
    if not LIBRARY_FOLDER.is_dir():
        return found
    for path in LIBRARY_FOLDER.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    found.sort()
    return found


def _embeddings_of(path: Path) -> list:
    """1ファイル内のすべての顔の特徴。ファイルが変わらなければキャッシュをそのまま使う。

    読めない場合や顔が1枚も検出されない場合は、エラーではなく空リストを返す。壊れた1枚で顔ライブラリ全体が
    壊れないようにしている。
    """
    try:
        modified_time = path.stat().st_mtime
    except OSError:
        return []          # スキャン直後に削除された

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
    """顔ライブラリの各顔について (名前のリスト, 特徴行列, 相対パスのリスト) を返す。

    相対パスは各顔がライブラリ内のどの写真に由来するかを表す位置で、1行が1つの名前に対応し、順序も
    一致する。これは結果行がその写真を指せるようにするためだけにあり、絶対パスが外部へ漏れることはない。
    """
    images = library_images()
    labels = []
    embeddings = []
    relative_paths = []
    alive = []

    # 1度に1リクエストだけライブラリを走査させる：ONNX セッションと検出ネットワークは2つのリクエストの重なりを嫌う。
    with LIBRARY_LOCK:
        for index, path in enumerate(images, start=1):
            if len(images) >= 20:
                print(f"  顔ライブラリ読み込み中 {index}/{len(images)}  {path.name}", flush=True)

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

        # 顔ライブラリから削除された画像のキャッシュも一緒に消す。
        for key in list(LIBRARY_CACHE):
            if key not in alive:
                del LIBRARY_CACHE[key]

    if not embeddings:
        return labels, np.zeros((0, 512), dtype="float32"), relative_paths
    return labels, np.vstack(embeddings).astype("float32"), relative_paths


def load_library() -> tuple:
    """顔ライブラリの各顔について (名前のリスト, 特徴行列) を返す。

    名前はファイル名。1枚の写真に複数の顔があるときは後ろに " #2"、" #3" を付ける。特徴行列の
    1行が1つの名前に対応し、順序も一致する。
    """
    labels, matrix, _relative_paths = _load_library_full()
    return labels, matrix


def library_image_url(relative_path: str) -> str:
    """顔ライブラリの写真1枚のブラウザ用アドレス。

    パスはエスケープが必要：サブフォルダ名に空白や日本語が含まれる可能性がある。フォルダ名として合法で
    URL としても合法だが、まずエスケープが必要になる。
    """
    return f"{LIBRARY_IMAGE_ROUTE}/{quote(relative_path)}"


# =============================================================================
# スコアを結果表に変換
# =============================================================================


def rank_matches(
    query_faces: list,
    labels: list,
    candidate_faces: Sequence,
    group_label: str,
    top_results: int,
    image_paths: Optional[Sequence] = None,
) -> list:
    """各候補をクエリ画像と1回ずつ比較し、上位数件だけ残す。

    labels は各候補の表示名で、順序は candidate_faces と一致する。group_label は
    候補の出所を示し、各行の "kind" に表示される（アップロードしたのは"候補画像"、
    顔ライブラリ内のものは"ライブラリ内の候補"）。

    返される行は {"rank", "kind", "candidate", "similarity"} で、1位にはさらに
    "best": True が付く。候補がなければ空リストを返す。

    image_paths を渡す場合は各名前に対応するライブラリ内の写真で、順序も一致し、各行に
    サムネイルのアドレスが1つ多く付く。渡されていない行（例：今回アップロードした候補画像）は、もともと表示する画像がない。
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
    """同名ファイルの後ろに番号を付け、結果表の2行が同じ見た目にならないようにする。"""
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
# Web ページ
# =============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


class RequestError(Exception):
    """このリクエストに問題があることを表す例外。メッセージはユーザー向け。"""


def error_response(message: str, status: int = 400):
    """ページと直接 API を叩く人が見るのと同じエラー形式。"""
    return jsonify({"ok": False, "error": message}), status


@app.errorhandler(RequestError)
def handle_request_error(error: RequestError):
    return error_response(str(error))


@app.errorhandler(413)
def handle_too_large(_error):
    return error_response("アップロードファイルが大きすぎます。1リクエストあたりの上限は 64 MB です", 413)


def read_uploads(field_name: str) -> list:
    """あるフィールドにアップロードされたすべてのファイル (ファイル名, バイト内容) を返す。"""
    uploads = []
    for item in request.files.getlist(field_name):
        if item and item.filename:
            uploads.append((item.filename, item.read()))
    return uploads


def encode_upload(data: bytes, label: str, file_name: str) -> list:
    """アップロード画像1枚に含まれるすべての顔の特徴を返す。

    画像が使えない場合（デコード失敗、破損、顔なし）は直接 RequestError を投げ、メッセージには
    どの画像かを入れ、label は"クエリ画像"または"候補画像"である。
    """
    try:
        faces = encode_faces(data)
    except (OSError, ValueError) as error:
        raise RequestError(f"{label}処理失敗：{error}：{file_name}")
    if not faces:
        raise RequestError(f"{label}顔が検出されません：{file_name}")
    return faces


def query_from_request() -> tuple:
    """今回のリクエストのクエリ画像 (ファイル名, すべての顔の特徴) を返す。"""
    uploads = read_uploads("query")
    if not uploads:
        raise RequestError("クエリ画像を選択してください")
    if len(uploads) > 1:
        raise RequestError("クエリ画像は1枚だけ選択できます")

    name, data = uploads[0]
    return name, encode_upload(data, "クエリ画像", name)


def top_results_from_request() -> int:
    """返す上位件数を求める。範囲外は境界値で計算する。"""
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


# -------------------- ページ --------------------
PAGE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>顔の類似度</title>
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
      <h1 class="h3 mb-1" style="color:#7a2e0a">顔の類似度</h1>
      <div class="text-secondary">アップロード比較 · 顔ライブラリ検索</div>
    </div>
    <div class="text-end">
      <span class="badge rounded-pill text-bg-light border" id="libraryBadge">顔ライブラリ読み込み中…</span>
    </div>
  </div>

  <div class="row g-3 align-items-stretch">
    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">1</span>クエリ画像 vs 候補画像（1:N）</div>
          <p class="form-text mb-3">クエリ画像1枚を、下で選んだ候補画像と1つずつ照合します。<strong>ローカル顔ライブラリは使いません</strong>。</p>
          <form id="queryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="queryFile">① クエリ画像（1枚）</label>
              <input class="form-control form-control-lg" type="file" id="queryFile"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="queryFile">例：誰かを検索.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="queryFile"></div>
            </div>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="candidateFiles">② 候補画像（複数選択可）</label>
              <input class="form-control form-control-lg" type="file" id="candidateFiles"
                     name="candidates" accept="image/*" multiple required>
              <div class="form-text" data-summary="candidateFiles">例：ユーザー1-1.jpg、ユーザー2-2.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="candidateFiles"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="queryTop">返す件数</label>
              <input class="form-control form-control-lg" type="number" id="queryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="querySubmit">
              比較開始
            </button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-6">
      <div class="panel">
        <div class="panel-body">
          <div class="panel-title mb-1"><span class="step">2</span>クエリ画像 vs 顔ライブラリ（1:N）</div>
          <p class="form-text mb-3">クエリ画像1枚と顔ライブラリ内の<b>すべて</b>の画像を比べ、似ている順に上位を自動で返します。</p>
          <form id="libraryForm" novalidate>
            <div class="mb-3">
              <label class="form-label fw-semibold" for="libraryQuery">① クエリ画像（1枚）</label>
              <input class="form-control form-control-lg" type="file" id="libraryQuery"
                     name="query" accept="image/*" required>
              <div class="form-text" data-summary="libraryQuery">例：誰かを検索.jpg</div>
              <div class="d-flex flex-wrap gap-2 mt-2" data-preview="libraryQuery"></div>
            </div>
            <div class="mb-4" style="max-width: 12rem">
              <label class="form-label fw-semibold" for="libraryTop">返す件数</label>
              <input class="form-control form-control-lg" type="number" id="libraryTop"
                     name="top_results" value="{{ top_results }}" min="{{ top_results_min }}" max="{{ top_results_max }}">
            </div>
            <button class="btn btn-primary btn-lg w-100" type="submit" id="librarySubmit">
              顔ライブラリ検索
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
            <div class="panel-title mb-0"><span class="step">3</span>結果</div>
            <div class="form-text" id="resultMeta"></div>
          </div>
          <div id="resultAlert"></div>
          <div id="resultBody" class="empty-state">まだ比較していません。</div>
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
    ? '<span class="spinner me-2"></span>比較中…'
    : idleText;
}

function showError(message) {
  $("resultAlert").innerHTML =
    '<div class="alert alert-brand alert-dismissible fade show" role="alert">' +
    '<strong>エラー：</strong>' + escapeHtml(message) +
    '<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>';
  $("resultBody").innerHTML = "";
  $("resultMeta").textContent = "";
}

function resultTable(rows) {
  if (!rows || !rows.length) {
    return '<p class="empty-state mb-0">比較できる対象がありません。</p>';
  }
  // The bar length is the similarity itself on a fixed 0-100 scale, so it
  // always matches the percentage printed next to it.
  const body = rows.map((row) => {
    const span = Math.max(0, Math.min(100, row.similarity));
    // image_url がない行は今回アップロードした候補画像で、ライブラリ内の写真ではないので表示する画像がない。
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
    <thead><tr><th>順位</th><th></th><th>${escapeHtml(rows[0].kind || "候補")}</th><th>類似度</th><th></th></tr></thead>
    <tbody>${body}</tbody></table>
    <p class="form-text mt-2 mb-0">横棒の長さは類似度の百分率で、バーいっぱいが 100%。</p>`;
}

function skippedList(skipped) {
  if (!skipped || !skipped.length) return "";
  return '<p class="form-text mt-3 mb-0">スキップ：'
    + skipped.map((item) => escapeHtml(item.name) + "（" + escapeHtml(item.status) + "）").join("、")
    + '</p>';
}

function verdictBlock(matches) {
  if (!matches || !matches.length) {
    return '<p class="empty-state">使える候補がありません。</p>';
  }
  const best = matches[0];
  return `<div class="verdict d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div>
        <div class="form-text mb-1">最も似ているのは</div>
        <div class="who">${escapeHtml(best.candidate)}</div>
      </div>
      <div class="text-end">
        <div class="form-text mb-1">類似度</div>
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
  if (payload.query) meta.push("クエリ：" + payload.query);
  if (payload.elapsed_ms != null) meta.push("総所要時間 " + payload.elapsed_ms + " ms");
  if (payload.count != null) meta.push("処理済み " + payload.count + " 枚");
  if (payload.library_faces != null) meta.push("顔ライブラリ " + payload.library_faces + " 枚の顔");
  if (payload.library_images != null) meta.push("顔ライブラリ " + payload.library_images + " 枚の画像");
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
      throw new Error("サーバーが解析できない内容を返しました（HTTP " + response.status + "）");
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || ("リクエスト失敗（HTTP " + response.status + "）"));
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
        showError("このパネルで必要な画像を先に選択してください");
        return;
      }
    }
    postJson(url, new FormData(form), $(buttonId), idleText);
  });
}

bindForm("queryForm", "querySubmit", "/api/query-set", "比較開始",
         ["queryFile", "candidateFiles"]);
bindForm("libraryForm", "librarySubmit", "/api/query-library", "顔ライブラリ検索",
         ["libraryQuery"]);

document.querySelectorAll('input[type="file"]').forEach((input) => {
  input.addEventListener("change", () => {
    const picked = Array.from(input.files);
    const summary = document.querySelector('[data-summary="' + input.id + '"]');
    if (summary && input.multiple) {
      summary.textContent = picked.length
        ? "選択済み " + picked.length + " 枚の画像"
        : "まだ選択していません";
    } else if (summary && picked.length) {
      summary.textContent = "選択：" + picked[0].name;
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
        ? "顔ライブラリ " + payload.faces + " 枚の顔 / " + payload.images + " 枚の画像"
        : "顔ライブラリは利用できません";
    })
    .catch(() => { $("libraryBadge").textContent = "顔ライブラリの状態が不明です"; });
}

loadLibraryStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    """ページ自体に2つのパネルがある。このページには検出方式のドロップダウンがない。

    3つの数字は、ページ上の"上位何件を表示"という入力欄の初期値と上下限を、
    top_results_from_request 内の境界と一致させる。
    """
    return render_template_string(PAGE, top_results=10, top_results_min=1, top_results_max=100)


@app.get("/api/library")
def api_library():
    """顔ライブラリに何枚の画像と何枚の顔があるか。

    数字だけ返し、フォルダパス自体はサーバーから出ない。
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
    """スクリプトとプロセスマネージャ用の"準備はできましたか"。

    わざと軽くしている：顔ライブラリ内の画像を数えるだけで特徴抽出は行わない。プローブは数秒ごとに
    1回問い合わせても、顔検出を1回も消費しない。"ready" はフォルダに1枚以上の画像があることを示すが、
    その画像に顔が含まれるかどうかは、実際に検索したときに判明する。
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
    """このマシンでサービスが現在実際に使っている設定。

    スコアを決める設定（モデル、特徴次元、閾値の方向、ライブラリのサイズ）とバージョン番号。
    2台のマシンで結果が違い、その理由を知りたいときはこれを見れば十分。顔ライブラリフォルダ
    については名前だけを報告し、パスは報告しない。
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
    """顔ライブラリ内の写真1枚。結果表のサムネイル用に、ブラウザから取れる URL を返す。

    send_from_directory がパスを顔ライブラリフォルダ内に固定し、外へ出ようとする
    あらゆる書き方を拒否するので、どんなパス指定でもマシン上の他のファイルには触れない。
    """
    return send_from_directory(LIBRARY_FOLDER, relative_path)


@app.get(LIBRARY_IMAGE_DOWNLOAD_ROUTE + "/<path:relative_path>")
def library_image_download(relative_path: str):
    """同じ写真をダウンロードする形式。特定の行を保存したり、個別に開いたりするのに便利。"""
    return send_from_directory(LIBRARY_FOLDER, relative_path, as_attachment=True)


@app.post("/api/query-set")
def api_query_set():
    """パネル1：クエリ画像を今回一緒にアップロードした候補画像と比較する。"""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    uploads = read_uploads("candidates")
    if not uploads:
        raise RequestError("候補画像を1枚以上選択してください")

    # 1枚が使えなければ記録してスキップ、全体の比較には影響なし。
    labels = []
    candidate_faces = []
    skipped = []
    for name, data in uploads:
        try:
            faces = encode_upload(data, "候補画像", name)
        except RequestError as error:
            skipped.append({"name": name, "status": str(error)})
            continue
        labels.append(name)
        candidate_faces.append(faces[0])        # この写真で最も大きい顔

    if not candidate_faces:
        raise RequestError("候補画像に使える顔がありません")

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
                "候補画像",
                top_results_from_request(),
            ),
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


@app.post("/api/query-library")
def api_query_library():
    """パネル2：クエリ画像を顔ライブラリ内の各画像と比較する。"""
    import time

    started = time.perf_counter()
    query_name, query_faces = query_from_request()

    if not library_images():
        raise RequestError("顔ライブラリに画像がありません。先に画像を入れてから再試行してください")

    labels, embedding_matrix, image_paths = _load_library_full()
    if embedding_matrix.shape[0] == 0:
        raise RequestError("顔ライブラリで顔が検出されませんでした。顔入りの画像を入れてから再試行してください")

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
                "ライブラリ内の候補",
                top_results_from_request(),
                image_paths,
            ),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    )


# =============================================================================
# 起動
# =============================================================================


def ask_library_folder() -> Path:
    """起動時に顔ライブラリの場所を一度だけ尋ねる。そのまま Enter ならスクリプトと同じ階層の face_library。

    尋ねられない場合（ダブルクリックで起動したときや他プログラムから起動したとき）は聞かず、既定の
    位置をそのまま使い、すでに閉じている入力ストリームは読まない。
    """
    default = SCRIPT_DIRECTORY / "face_library"
    if not sys.stdin.isatty():
        default.mkdir(parents=True, exist_ok=True)
        return default

    print("顔ライブラリは画像を置くフォルダで、サブディレクトリ可。")
    print(f"対応フォーマット：{' '.join(sorted(SUPPORTED_EXTENSIONS))}")
    while True:
        answer = input(f"顔ライブラリフォルダ（直接 Enter で {default}）：").strip()
        if not answer:
            default.mkdir(parents=True, exist_ok=True)
            return default
        folder = Path(answer).expanduser()
        if folder.is_dir():
            return folder
        if folder.exists():
            print("それはファイルであってフォルダではありません。再入力してください。")
        else:
            print("このフォルダは存在しません。パスを確認して再入力してください。")


def choose_port(preferred: int = 5000) -> int:
    """preferred から後ろへ順に空きポートを探す。

    5000 はよく使われているので、使われていれば自動で後ろへずらして起動失敗を避ける。
    """
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # ここでは SO_REUSEADDR を設定しない：Windows 上で設定すると、このプローブが別の
            # プログラムが待ち受けているポートに結びつく。
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"{preferred} ~ {preferred + 19} 間に空きポートなし")


def main():
    """顔ライブラリの場所を確認し、モデルを読み込んでから Web サービスを起動する。"""
    global LIBRARY_FOLDER
    LIBRARY_FOLDER = ask_library_folder()

    print("読み込み中 buffalo_l モデル、約2秒…", flush=True)
    load_model()
    print("モデル読み込み完了")
    print(MODEL_SUMMARY, flush=True)
    print(f"顔ライブラリディレクトリ: {LIBRARY_FOLDER}", flush=True)

    images = library_images()
    print(f"現在の顔ライブラリには {len(images)} 枚の画像", flush=True)

    port = choose_port()
    print(f"Open http://127.0.0.1:{port} in your browser", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()