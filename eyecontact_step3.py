"""
eyecontact_step3.py  —  POV 第三步子步驟:眼神接觸偵測
----------------------------------------------------------------
核心簡化:智能眼鏡鏡頭 ≈ 治療師眼睛,所以「兒童看鏡頭」≈「與治療師眼神接觸」。
於是問題從困難的 3D 視線估計,簡化成「這張臉有沒有正對鏡頭 + 眼睛有沒有看向鏡頭」。

判斷依據(用 MediaPipe Face Landmarker,你機器上已裝好的那套):
  1. 頭部姿態 yaw / pitch(臉夠不夠正對鏡頭)—— 由臉部變換矩陣取得
  2. 眼睛偏移 eye_away(眼球有沒有偏離中心)—— 由 blendshape 取得
  兩者都在門檻內才判定為「看鏡頭」。

沿用架構:讀第一步的 detections CSV(已有框/時間戳/face_id)→ 逐臉判斷 →
輸出帶標記的 CSV。不重跑偵測。

安裝(本機):
    py -m pip install mediapipe opencv-python-headless numpy
第一次執行自動下載模型 face_landmarker.task。

用法:
    py eyecontact_step3.py <影片> <detections.csv> [-o 前綴]
        [--yaw 20] [--pitch 15] [--eye 0.3] [--pad 0.3]

範例:
    py eyecontact_step3.py barber.mp4 barber_mp_detections.csv -o barber_eye
"""
import cv2, csv, os, argparse, math, urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_FILE = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
EYE_BLENDSHAPES = ["eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
                   "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
                   "eyeLookDownLeft", "eyeLookDownRight"]

def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"第一次執行,下載臉部關鍵點模型 {path} ...")
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        if size < 1_000_000:
            raise SystemExit(f"模型檔異常小({size} bytes),下載可能失敗。手動下載:\n{url}")
        print(f"模型下載完成({size/1e6:.1f} MB)。")

def euler_from_R(R):
    """3x3 旋轉矩陣 -> (pitch, yaw, roll) 度。"""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1]); yaw = math.atan2(-R[2, 0], sy); roll = 0.0
    return [math.degrees(a) for a in (pitch, yaw, roll)]

def make_landmarker():
    base = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True)
    return mp_vision.FaceLandmarker.create_from_options(opts)

def analyze_face(landmarker, face_bgr):
    """回傳 (yaw, pitch, roll, eye_away) 或 None(沒找到臉)。"""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.facial_transformation_matrixes:
        return None
    M = np.array(res.facial_transformation_matrixes[0])
    pitch, yaw, roll = euler_from_R(M[:3, :3])
    eye_away = 0.0
    if res.face_blendshapes:
        scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
        eye_away = max((scores.get(n, 0.0) for n in EYE_BLENDSHAPES), default=0.0)
    return yaw, pitch, roll, eye_away

def crop(frame, x, y, w, h, pad):
    H, W = frame.shape[:2]
    px, py = int(w*pad), int(h*pad)
    x0, y0 = max(0, x-px), max(0, y-py)
    x1, y1 = min(W, x+w+px), min(H, y+h+py)
    return frame[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None

def color_for(tid):
    rng = np.random.default_rng(int(tid) * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("detections_csv")
    ap.add_argument("-o", "--output")
    ap.add_argument("--yaw", type=float, default=20.0, help="yaw 門檻(度),超過視為沒正對")
    ap.add_argument("--pitch", type=float, default=15.0, help="pitch 門檻(度)")
    ap.add_argument("--eye", type=float, default=0.3, help="眼睛偏移門檻(0~1),超過視為看別處")
    ap.add_argument("--pad", type=float, default=0.3, help="裁臉留邊比例(關鍵點需要多一點臉)")
    args = ap.parse_args()

    ensure_model(MODEL_FILE, MODEL_URL)
    prefix = args.output or os.path.splitext(os.path.basename(args.detections_csv))[0]
    out_csv, out_video = f"{prefix}_eyecontact.csv", f"{prefix}_eyecontact.mp4"

    with open(args.detections_csv, newline="") as f:
        reader = csv.reader(f); header = next(reader)
        ts_col = header[0]; rows = list(reader)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r[1]), []).append(r)
    if not by_frame:
        raise SystemExit("detections CSV 沒有資料。")
    fmin, fmax = min(by_frame), max(by_frame)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不開影片: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    landmarker = make_landmarker()

    cap.set(cv2.CAP_PROP_POS_FRAMES, fmin)
    out_rows, idx = [], fmin
    n_faces = n_look = n_nolandmark = 0
    per_id = {}     # face_id -> [看鏡頭幀數, 總幀數]
    while idx <= fmax:
        ok, frame = cap.read()
        if not ok: break
        if idx in by_frame:
            for r in by_frame[idx]:
                ts, fidx, fid = r[0], int(r[1]), int(r[2])
                x, y, w, h = map(int, r[3:7])
                face = crop(frame, x, y, w, h, args.pad)
                per_id.setdefault(fid, [0, 0])
                per_id[fid][1] += 1
                n_faces += 1
                res = analyze_face(landmarker, face) if face is not None else None
                if res is None:
                    n_nolandmark += 1
                    out_rows.append([ts, fidx, fid, x, y, w, h, "", "", "", "", 0])
                    c = (120, 120, 120)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), c, 1)
                    idx_label = f"id{fid} no-landmark"
                    cv2.putText(frame, idx_label, (x, max(15, y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                    continue
                yaw, pitch, roll, eye_away = res
                looking = (abs(yaw) <= args.yaw and abs(pitch) <= args.pitch
                           and eye_away <= args.eye)
                if looking:
                    n_look += 1; per_id[fid][0] += 1
                out_rows.append([ts, fidx, fid, x, y, w, h,
                                 f"{yaw:.1f}", f"{pitch:.1f}", f"{roll:.1f}",
                                 f"{eye_away:.3f}", int(looking)])
                c = (0, 220, 0) if looking else color_for(fid)
                label = "EYE CONTACT" if looking else "no"
                cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2 if looking else 1)
                cv2.putText(frame, f"id{fid} {label}", (x, max(15, y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                cv2.putText(frame, f"y{yaw:.0f} p{pitch:.0f} e{eye_away:.2f}",
                            (x, y+h+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        cv2.putText(frame, f"frame {idx}", (10, H-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        vw.write(frame); idx += 1

    cap.release(); vw.release(); landmarker.close()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([ts_col, "frame_idx", "face_id", "x", "y", "w", "h",
                    "yaw", "pitch", "roll", "eye_away", "looking_at_camera"])
        w.writerows(out_rows)

    analyzable = n_faces - n_nolandmark
    print(f"影片            : {args.video}")
    print(f"門檻            : |yaw|<={args.yaw}  |pitch|<={args.pitch}  eye_away<={args.eye}")
    print(f"讀入偵測         : {len(rows)} 筆(frame {fmin}~{fmax})")
    print(f"可分析臉數       : {analyzable}(另有 {n_nolandmark} 張抓不到關鍵點,多為側臉/遮擋)")
    if analyzable:
        print(f"判定看鏡頭       : {n_look} ({n_look/analyzable*100:.1f}% of 可分析臉)")
    print("\n各 face_id 的眼神接觸比例(臨床上的『注意力/互動』指標雛形):")
    for fid, (look, tot) in sorted(per_id.items()):
        print(f"  id{fid}: {look}/{tot} 幀看鏡頭 = {look/max(1,tot)*100:.1f}%")
    print(f"\n輸出: {out_csv}\n輸出: {out_video}")

if __name__ == "__main__":
    main()