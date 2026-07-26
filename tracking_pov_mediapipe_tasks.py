"""
tracking_pov_mediapipe.py  —  POV 臉部偵測(MediaPipe Tasks API)+ 追蹤
--------------------------------------------------------------------------------
相容新版 mediapipe(0.10.30+,已移除舊的 mp.solutions)。
介面/輸出與舊版完全相同,只是改用新版 Tasks API 的 FaceDetector。

安裝(本機執行):
    py -m pip install mediapipe opencv-python-headless numpy

第一次執行會自動下載模型檔 blaze_face_short_range.tflite(需連外網,只下載一次)。

用法:
    py tracking_pov_mediapipe.py <影片路徑> [-o 輸出前綴] [--start T] [--end T] [--conf 0.5]

時間 T 可用秒數或時間碼: 90 / 1:30 / 01:02:03
範例:
    py tracking_pov_mediapipe.py pov_clip.mp4 -o seg1_mp --start 1:20 --end 1:50
"""
import cv2, csv, os, argparse, urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_FILE = "blaze_face_short_range.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_detector/blaze_face_short_range/float16/1/"
             "blaze_face_short_range.tflite")

def ensure_model():
    """模型檔不存在就自動下載(只做一次)。"""
    if not os.path.exists(MODEL_FILE):
        print(f"第一次執行,下載模型檔 {MODEL_FILE} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("模型下載完成。")

def make_detector(conf):
    base = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    opts = mp_vision.FaceDetectorOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=conf)
    return mp_vision.FaceDetector.create_from_options(opts)

def detect_faces(detector, frame_bgr, ts_ms):
    """回傳 [(x,y,w,h), ...],座標為像素並夾在畫面內。"""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect_for_video(mp_img, int(ts_ms))
    boxes = []
    for d in res.detections:
        bb = d.bounding_box
        x, y = max(0, bb.origin_x), max(0, bb.origin_y)
        bw, bh = min(bb.width, w - x), min(bb.height, h - y)
        if bw > 0 and bh > 0:
            boxes.append((x, y, bw, bh))
    return boxes

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0

class IOUTracker:
    def __init__(self, iou_thr=0.3, max_age=15):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks, self.next_id = {}, 1
    def update(self, dets):
        for t in self.tracks.values(): t["age"] += 1
        mt, md, pairs = set(), set(), []
        for tid, t in self.tracks.items():
            for di, d in enumerate(dets):
                pairs.append((iou(t["bbox"], d), tid, di))
        for s, tid, di in sorted(pairs, reverse=True):
            if s < self.iou_thr: break
            if tid in mt or di in md: continue
            self.tracks[tid]["bbox"] = dets[di]; self.tracks[tid]["age"] = 0
            mt.add(tid); md.add(di)
        for di, d in enumerate(dets):
            if di in md: continue
            self.tracks[self.next_id] = {"bbox": d, "age": 0}
            md.add(di); self.next_id += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}
        return [(tid, t["bbox"]) for tid, t in self.tracks.items() if t["age"] == 0]

def color_for(tid):
    rng = np.random.default_rng(tid * 9973)
    return tuple(int(c) for c in rng.integers(60, 255, 3))

def parse_time(s):
    if s is None: return None
    parts = str(s).split(":")
    if len(parts) == 1: return float(parts[0])
    if len(parts) == 2: return int(parts[0])*60 + float(parts[1])
    if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
    raise ValueError(f"時間格式看不懂: {s}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="影片路徑")
    ap.add_argument("-o", "--output", help="輸出前綴(預設用檔名)")
    ap.add_argument("--start", help="起始時間 (秒 或 MM:SS / HH:MM:SS)")
    ap.add_argument("--end", help="結束時間 (秒 或 MM:SS / HH:MM:SS)")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="偵測信心門檻 0~1(預設 0.5;想少漏調低如 0.3,想少誤報調高)")
    args = ap.parse_args()

    ensure_model()
    start_s = parse_time(args.start)
    end_s = parse_time(args.end)
    prefix = args.output or os.path.splitext(os.path.basename(args.video))[0]
    out_csv, out_video = f"{prefix}_detections.csv", f"{prefix}_tracked.mp4"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不開影片: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tracker = IOUTracker()
    detector = make_detector(args.conf)

    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    rows, n_out, frames_with_face = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = ms/1000.0 if ms and ms > 0 else None
        if ts is None:
            ts = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            ms = ts * 1000.0
        if start_s and ts < start_s - 1e-3:
            continue
        if end_s and ts > end_s:
            break

        frame_idx = int(round(ts * fps))
        # detect_for_video 需要單調遞增的整數毫秒時間戳
        tracked = tracker.update(detect_faces(detector, frame, ms))
        if tracked: frames_with_face += 1
        for tid, (x, y, w, h) in tracked:
            rows.append([f"{ts:.6f}", frame_idx, tid, x, y, w, h])
            c = color_for(tid)
            cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
            cv2.putText(frame, f"face {tid}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(frame, f"t={ts:.3f}s", (10, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        vw.write(frame); n_out += 1

    cap.release(); vw.release()
    detector.close()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_seconds", "frame_idx", "face_id", "x", "y", "w", "h"])
        w.writerows(rows)

    ids = sorted({r[2] for r in rows})
    seg = f"[{args.start or '開頭'} -> {args.end or '結尾'}]"
    print(f"影片            : {args.video}  ({W}x{H}, {fps:.2f}fps)")
    print(f"偵測器          : MediaPipe Tasks FaceDetector (conf={args.conf})")
    print(f"處理區間        : {seg}")
    print(f"處理幀數        : {n_out}")
    print(f"有偵到臉的幀數  : {frames_with_face} ({(frames_with_face/max(1,n_out))*100:.1f}%)")
    print(f"總偵測筆數      : {len(rows)}")
    print(f"不同 face_id 數 : {len(ids)}")
    print(f"輸出: {out_csv}\n輸出: {out_video}")

if __name__ == "__main__":
    main()