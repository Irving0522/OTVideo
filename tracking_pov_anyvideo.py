"""
tracking_pov_anyvideo.py  —  吃任何 mp4 的 POV 臉部偵測 + 追蹤(可指定時間區間)
--------------------------------------------------------------------------------
時間戳直接從影片本身讀,不需外部 CSV。可只跑影片的某一段(適合長片剪出有人臉的片段)。

用法:
    python tracking_pov_anyvideo.py <影片路徑> [-o 輸出前綴] [--start T] [--end T]

時間 T 可用秒數或時間碼:  90  /  1:30  /  01:30  /  1:02:03
範例:
    python tracking_pov_anyvideo.py pov_clip.mp4 --start 1:20 --end 1:50
    python tracking_pov_anyvideo.py pov_clip.mp4 -o seg1 --start 80 --end 110

換成真 Neon 影片時:把 timestamp 來源從『影片內建 ms』換成 Neon 匯出的
硬體 timestamp_unix_seconds,其餘不動——這樣兩視角才能精確對齊。
"""
import cv2, csv, os, argparse
import numpy as np

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(45, 45))
    return [tuple(map(int, f)) for f in faces]

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
    """'90' / '1:30' / '01:02:03' -> 秒(float)。"""
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
    args = ap.parse_args()

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
        if start_s and ts < start_s - 1e-3:
            continue
        if end_s and ts > end_s:
            break

        frame_idx = int(round(ts * fps))
        tracked = tracker.update(detect_faces(frame))
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
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_seconds", "frame_idx", "face_id", "x", "y", "w", "h"])
        w.writerows(rows)

    ids = sorted({r[2] for r in rows})
    seg = f"[{args.start or '開頭'} -> {args.end or '結尾'}]"
    print(f"影片            : {args.video}  ({W}x{H}, {fps:.2f}fps)")
    print(f"處理區間        : {seg}")
    print(f"處理幀數        : {n_out}")
    print(f"有偵到臉的幀數  : {frames_with_face} ({(frames_with_face/max(1,n_out))*100:.1f}%)")
    print(f"總偵測筆數      : {len(rows)}")
    print(f"出現過的 face_id: {ids}")
    print(f"輸出: {out_csv}\n輸出: {out_video}")

if __name__ == "__main__":
    main()