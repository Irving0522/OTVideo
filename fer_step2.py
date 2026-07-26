"""
fer_step2.py  —  POV 第二步子步驟:情緒辨識(FER)
----------------------------------------------------------------
沿用「分析層不重跑偵測模型」的架構:讀第一步產出的 detections CSV(裡面已有
每張臉的框 + 時間戳 + face_id),把臉切出來丟進情緒模型,輸出加了情緒欄位的新 CSV。

模型:ONNX emotion-ferplus(輕量,8 類情緒)。第一次執行自動下載模型檔。
8 類:neutral / happiness / surprise / sadness / anger / disgust / fear / contempt

安裝(本機):
    py -m pip install onnxruntime opencv-python-headless numpy

用法:
    py fer_step2.py <影片> <detections.csv> [-o 輸出前綴] [--pad 0.2] [--min-score 0.0]

範例:
    py fer_step2.py barber.mp4 barber_mp_detections.csv -o barber_emo

輸出:
    <前綴>_emotions.csv   在 detections 每一列後面加上 emotion + 各類分數
    <前綴>_emotions.mp4   標註影片(框 + face_id + 情緒)
"""
import cv2, csv, os, argparse, urllib.request
import numpy as np
import onnxruntime as ort

MODEL_FILE = "emotion-ferplus-8.onnx"
MODEL_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
             "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
EMOTIONS = ["neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"]

def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"第一次執行,下載情緒模型 {path} ...")
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        if size < 1_000_000:      # 正常應該有幾十 MB;太小代表抓到的是 LFS 指標檔
            raise SystemExit(
                f"模型檔異常小({size} bytes),可能下載失敗。\n"
                f"請手動下載後放到同資料夾:\n  {url}")
        print(f"模型下載完成({size/1e6:.1f} MB)。")

def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()

def preprocess(face_bgr):
    """emotion-ferplus 前處理:灰階 -> 64x64 -> float32 (0~255) -> (1,1,64,64)。"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)).astype(np.float32)
    return resized.reshape(1, 1, 64, 64)

def crop(frame, x, y, w, h, pad):
    """依框裁臉,四周加 pad 比例的邊(FER 對含一點下巴/額頭的臉較準)。"""
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
    ap.add_argument("--pad", type=float, default=0.2, help="裁臉四周留邊比例(預設 0.2)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="情緒信心低於此值標為 uncertain(預設 0,不過濾)")
    ap.add_argument("--model", default=MODEL_FILE)
    args = ap.parse_args()

    ensure_model(args.model, MODEL_URL)
    prefix = args.output or os.path.splitext(os.path.basename(args.detections_csv))[0]
    out_csv, out_video = f"{prefix}_emotions.csv", f"{prefix}_emotions.mp4"

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name          # 動態讀名稱,不寫死
    out_name = sess.get_outputs()[0].name

    # 讀 detections,依 frame_idx 分組
    with open(args.detections_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ts_col = header[0]                        # 第一欄是時間戳(名稱可能不同)
        rows = list(reader)
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, fmin)        # 跳到第一個有偵測的幀
    out_rows = []
    emo_count = {e: 0 for e in EMOTIONS}
    processed_faces = 0
    idx = fmin
    while idx <= fmax:
        ok, frame = cap.read()
        if not ok: break
        if idx in by_frame:
            for r in by_frame[idx]:
                ts, fidx, fid = r[0], int(r[1]), int(r[2])
                x, y, w, h = map(int, r[3:7])
                face = crop(frame, x, y, w, h, args.pad)
                if face is None or face.size == 0:
                    continue
                logits = sess.run([out_name], {in_name: preprocess(face)})[0][0]
                probs = softmax(logits)
                top = int(np.argmax(probs))
                emo = EMOTIONS[top]
                score = float(probs[top])
                if score < args.min_score:
                    emo_label = "uncertain"
                else:
                    emo_label = emo
                    emo_count[emo] += 1
                processed_faces += 1
                out_rows.append([ts, fidx, fid, x, y, w, h,
                                 emo_label, f"{score:.4f}"] +
                                [f"{p:.4f}" for p in probs])
                # 標註
                c = color_for(fid)
                cv2.rectangle(frame, (x, y), (x+w, y+h), c, 2)
                cv2.putText(frame, f"id{fid} {emo_label} {score:.2f}",
                            (x, max(15, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.putText(frame, f"frame {idx}", (10, H-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        vw.write(frame)
        idx += 1

    cap.release(); vw.release()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([ts_col, "frame_idx", "face_id", "x", "y", "w", "h",
                    "emotion", "emotion_score"] + [f"p_{e}" for e in EMOTIONS])
        w.writerows(out_rows)

    print(f"影片            : {args.video}")
    print(f"讀入偵測         : {len(rows)} 筆(frame {fmin}~{fmax})")
    print(f"完成情緒判定     : {processed_faces} 張臉")
    print("情緒分布:")
    total = max(1, sum(emo_count.values()))
    for e in EMOTIONS:
        if emo_count[e]:
            print(f"  {e:10s}: {emo_count[e]:4d}  ({emo_count[e]/total*100:.1f}%)")
    print(f"\n輸出: {out_csv}\n輸出: {out_video}")

if __name__ == "__main__":
    main()