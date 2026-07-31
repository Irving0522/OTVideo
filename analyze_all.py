"""
analyze_all.py  —  一鍵分析:丟一支影片,直接輸出「已疊圖的成品影片」
================================================================================
一個指令跑完整條 POV 管線,並輸出一支可直接播放的成品影片:
  - 臉上即時顯示【情緒(中文)+ 眼神接觸】
  - 畫面下方顯示【時間條】
  - 保留原始聲音
同時存下各步驟的 CSV(偵測 / 情緒 / 眼神),並呼叫 speech_step4.py 做說話分析。

不需先產生 CSV、不需開網頁——丟影片就出結果。

安裝(本機):
    py -m pip install mediapipe onnxruntime opencv-python-headless numpy pillow imageio-ffmpeg faster-whisper
    (模型第一次執行自動下載;imageio-ffmpeg 用來接回音軌;faster-whisper 供說話分析)

用法:
    py analyze_all.py <影片> [-o 前綴] [--start T --end T]
        [--stride N] [--no-speech] [--conf 0.5] [--yaw 20 --pitch 15 --eye 0.3]

範例:
    py analyze_all.py interview.mp4 --start 1:00 --end 1:30
    → interview_analyzed.mp4(疊好情緒+眼神+時間條、含聲音)+ 各步 CSV
"""
import cv2, csv, os, sys, math, argparse, urllib.request, subprocess, shutil
import numpy as np

# ---------------- 模型檔(第一次執行自動下載)----------------
FD_MODEL = "blaze_face_short_range.tflite"
FD_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
          "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
EMO_MODEL = "emotion-ferplus-8.onnx"
EMO_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
           "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
LM_MODEL = "face_landmarker.task"
LM_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
          "face_landmarker/float16/1/face_landmarker.task")

EMOTIONS = ["neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"]
EMO_ZH = {"neutral": "中性", "happiness": "開心", "surprise": "驚訝",
          "sadness": "難過", "anger": "生氣", "disgust": "厭惡",
          "fear": "害怕", "contempt": "輕蔑", "uncertain": "不確定"}
EYE_BLENDSHAPES = ["eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
                   "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
                   "eyeLookDownLeft", "eyeLookDownRight"]
CJK_FONTS = [
    r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]

def ensure(path, url, min_mb):
    if not os.path.exists(path):
        print(f"下載模型 {path} ...")
        urllib.request.urlretrieve(url, path)
        if os.path.getsize(path) < min_mb*1_000_000*0.5:
            raise SystemExit(f"{path} 下載異常(檔案過小),請手動下載:\n{url}")
        print(f"  完成({os.path.getsize(path)/1e6:.1f} MB)")

def parse_time(s):
    if s is None: return None
    p = str(s).split(":")
    if len(p) == 1: return float(p[0])
    if len(p) == 2: return int(p[0])*60 + float(p[1])
    if len(p) == 3: return int(p[0])*3600 + int(p[1])*60 + float(p[2])
    raise ValueError(f"時間格式看不懂: {s}")

def fmt_ts(sec):
    m = int(sec//60); s = int(sec%60); return f"{m:02d}:{s:02d}"

# ---------------- 幾何/推論小工具 ----------------
def euler_from_R(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        return [math.degrees(math.atan2(R[2,1], R[2,2])),
                math.degrees(math.atan2(-R[2,0], sy)),
                math.degrees(math.atan2(R[1,0], R[0,0]))]
    return [math.degrees(math.atan2(-R[1,2], R[1,1])),
            math.degrees(math.atan2(-R[2,0], sy)), 0.0]

def softmax(v):
    e = np.exp(v - np.max(v)); return e/e.sum()

def iou(a, b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    x1,y1=max(ax,bx),max(ay,by); x2,y2=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    inter=max(0,x2-x1)*max(0,y2-y1); uni=aw*ah+bw*bh-inter
    return inter/uni if uni>0 else 0.0

class IOUTracker:
    def __init__(self, thr=0.3, max_age=15):
        self.thr=thr; self.max_age=max_age; self.tracks={}; self.next_id=1
    def update(self, dets):
        for t in self.tracks.values(): t["age"]+=1
        mt,md,pairs=set(),set(),[]
        for tid,t in self.tracks.items():
            for di,d in enumerate(dets): pairs.append((iou(t["bbox"],d),tid,di))
        for s,tid,di in sorted(pairs,reverse=True):
            if s<self.thr: break
            if tid in mt or di in md: continue
            self.tracks[tid]["bbox"]=dets[di]; self.tracks[tid]["age"]=0; mt.add(tid); md.add(di)
        for di,d in enumerate(dets):
            if di in md: continue
            self.tracks[self.next_id]={"bbox":d,"age":0}; md.add(di); self.next_id+=1
        self.tracks={k:v for k,v in self.tracks.items() if v["age"]<=self.max_age}
        return [(tid,t["bbox"]) for tid,t in self.tracks.items() if t["age"]==0]

def crop(frame,x,y,w,h,pad):
    H,W=frame.shape[:2]; px,py=int(w*pad),int(h*pad)
    x0,y0=max(0,x-px),max(0,y-py); x1,y1=min(W,x+w+px),min(H,y+h+py)
    return frame[y0:y1,x0:x1] if x1>x0 and y1>y0 else None

# ---------------- 疊圖繪製(PIL,支援中文)----------------
class Overlay:
    def __init__(self, W, H):
        from PIL import ImageFont
        self.PIL_ok=True
        base=max(14, int(H*0.032))
        self.fp=next((f for f in CJK_FONTS if os.path.exists(f)), None)
        try:
            self.font=ImageFont.truetype(self.fp, base) if self.fp else ImageFont.load_default()
            self.font_s=ImageFont.truetype(self.fp, int(base*0.8)) if self.fp else self.font
            self.cjk = self.fp is not None
        except Exception:
            self.font=ImageFont.load_default(); self.font_s=self.font; self.cjk=False
        self.base=base
    def _label(self, emotion, looking):
        emo = EMO_ZH.get(emotion, emotion) if self.cjk else emotion
        eye = ("看鏡頭" if self.cjk else "eye contact") if looking else ("沒看" if self.cjk else "no")
        return emo, eye
    def draw(self, frame_bgr, faces, t, total):
        from PIL import Image, ImageDraw
        img=Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        d=ImageDraw.Draw(img,"RGBA"); W,H=img.size
        for f in faces:
            x,y,w,h=f["box"]; look=f["looking"]
            col=(46,205,120) if look else (232,163,45)     # 綠=看鏡頭,琥珀=沒看
            d.rectangle([x,y,x+w,y+h], outline=col, width=max(2,int(H*0.004)))
            emo,eye=self._label(f["emotion"], look)
            txt=f"{emo}  ·  {eye}"
            tb=d.textbbox((0,0),txt,font=self.font); tw,th=tb[2]-tb[0],tb[3]-tb[1]
            pad=int(th*0.35); by=max(0,y-th-2*pad)
            d.rectangle([x,by,x+tw+2*pad,by+th+2*pad], fill=(col[0],col[1],col[2],220))
            d.text((x+pad,by+pad-tb[1]),txt,font=self.font,fill=(255,255,255))
            if f.get("emotion")=="uncertain":
                pass
        # 下方時間條
        barH=max(6,int(H*0.02)); m=int(W*0.03); y0=H-barH-int(H*0.03)
        d.rectangle([m,y0,W-m,y0+barH], fill=(255,255,255,60))
        prog=(t/total) if total else 0
        d.rectangle([m,y0,m+int((W-2*m)*prog),y0+barH], fill=(18,181,214,235))
        tt=f"{fmt_ts(t)} / {fmt_ts(total)}"
        tb=d.textbbox((0,0),tt,font=self.font_s)
        d.text((W-m-(tb[2]-tb[0]), y0-int(H*0.035)),tt,font=self.font_s,fill=(255,255,255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------------- 主流程 ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o","--output")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--stride",type=int,default=1,help="每 N 幀分析一次(略過的幀沿用上次結果加速),預設 1")
    ap.add_argument("--conf",type=float,default=0.5)
    ap.add_argument("--yaw",type=float,default=20.0)
    ap.add_argument("--pitch",type=float,default=15.0)
    ap.add_argument("--eye",type=float,default=0.3)
    ap.add_argument("--no-speech",action="store_true",help="略過說話分析")
    args=ap.parse_args()

    if not os.path.exists(args.video): sys.exit(f"找不到影片:{args.video}")
    prefix=args.output or os.path.splitext(os.path.basename(args.video))[0]
    final_mp4=f"{prefix}_analyzed.mp4"; tmp_mp4=f"{prefix}_analyzed_silent.mp4"

    ensure(FD_MODEL,FD_URL,0.2); ensure(EMO_MODEL,EMO_URL,30); ensure(LM_MODEL,LM_URL,3)

    import onnxruntime as ort
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv

    fd=mpv.FaceDetector.create_from_options(mpv.FaceDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=FD_MODEL),
        running_mode=mpv.RunningMode.VIDEO, min_detection_confidence=args.conf))
    emo_sess=ort.InferenceSession(EMO_MODEL, providers=["CPUExecutionProvider"])
    emo_in=emo_sess.get_inputs()[0].name; emo_out=emo_sess.get_outputs()[0].name
    lm=mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=LM_MODEL),
        running_mode=mpv.RunningMode.IMAGE, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit(f"打不開影片:{args.video}")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    start=parse_time(args.start) or 0.0
    total_dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    end=parse_time(args.end) or total_dur
    if start>0: cap.set(cv2.CAP_PROP_POS_MSEC, start*1000.0)

    vw=cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    ov=Overlay(W,H)
    tracker=IOUTracker()

    det_rows,emo_rows,eye_rows=[],[],[]
    emo_count={e:0 for e in EMOTIONS}; look_cnt=[0,0]  # [看鏡頭, 可分析]
    last_faces=[]; idx=0; processed=0
    seg_dur=end-start
    print(f"分析中… 影片 {W}x{H}@{fps:.0f}fps,區間 {fmt_ts(start)}–{fmt_ts(end)}")

    while True:
        ok,frame=cap.read()
        if not ok: break
        ms=cap.get(cv2.CAP_PROP_POS_MSEC); ts=ms/1000.0 if ms>0 else (start+idx/fps)
        if ts>end+1e-3: break
        rel=ts-start

        if idx % max(1,args.stride)==0:
            # 偵測
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=fd.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(ms) if ms>0 else idx)
            dets=[]
            for d in res.detections:
                bb=d.bounding_box; x,y=max(0,bb.origin_x),max(0,bb.origin_y)
                w,h=min(bb.width,W-x),min(bb.height,H-y)
                if w>0 and h>0: dets.append((x,y,w,h))
            tracked=tracker.update(dets)
            faces=[]
            for tid,(x,y,w,h) in tracked:
                fcrop=crop(frame,x,y,w,h,0.25)
                emotion,escore="uncertain",0.0
                if fcrop is not None and fcrop.size:
                    g=cv2.resize(cv2.cvtColor(fcrop,cv2.COLOR_BGR2GRAY),(64,64)).astype(np.float32).reshape(1,1,64,64)
                    probs=softmax(emo_sess.run([emo_out],{emo_in:g})[0][0])
                    k=int(np.argmax(probs)); emotion=EMOTIONS[k]; escore=float(probs[k])
                # 眼神
                yaw=pitch=eye_away=None; looking=False
                lcrop=crop(frame,x,y,w,h,0.3)
                if lcrop is not None and lcrop.size:
                    lr=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(lcrop,cv2.COLOR_BGR2RGB)))
                    if lr.facial_transformation_matrixes:
                        M=np.array(lr.facial_transformation_matrixes[0]); pitch,yaw,_=euler_from_R(M[:3,:3])
                        eye_away=0.0
                        if lr.face_blendshapes:
                            sc={c.category_name:c.score for c in lr.face_blendshapes[0]}
                            eye_away=max((sc.get(n,0.0) for n in EYE_BLENDSHAPES),default=0.0)
                        looking=(abs(yaw)<=args.yaw and abs(pitch)<=args.pitch and eye_away<=args.eye)
                        look_cnt[1]+=1
                        if looking: look_cnt[0]+=1
                faces.append({"box":(x,y,w,h),"id":tid,"emotion":emotion,"looking":looking})
                emo_count[emotion]=emo_count.get(emotion,0)+1
                det_rows.append([f"{ts:.6f}",idx,tid,x,y,w,h])
                emo_rows.append([f"{ts:.6f}",idx,tid,x,y,w,h,emotion,f"{escore:.4f}"])
                eye_rows.append([f"{ts:.6f}",idx,tid,x,y,w,h,
                                 "" if yaw is None else f"{yaw:.1f}",
                                 "" if pitch is None else f"{pitch:.1f}",
                                 "" if eye_away is None else f"{eye_away:.3f}",
                                 int(looking)])
            last_faces=faces; processed+=1
        # 疊圖(略過的幀沿用 last_faces)
        out=ov.draw(frame,last_faces,rel,seg_dur)
        vw.write(out); idx+=1
        if idx % 60==0: print(f"  ...{fmt_ts(rel)} / {fmt_ts(seg_dur)}")

    cap.release(); vw.release(); fd.close(); lm.close()

    # CSV
    def wr(fn,header,rows):
        with open(fn,"w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f); w.writerow(header); w.writerows(rows)
    wr(f"{prefix}_detections.csv",["timestamp_seconds","frame_idx","face_id","x","y","w","h"],det_rows)
    wr(f"{prefix}_emotions.csv",["timestamp_seconds","frame_idx","face_id","x","y","w","h","emotion","emotion_score"],emo_rows)
    wr(f"{prefix}_eyecontact.csv",["timestamp_seconds","frame_idx","face_id","x","y","w","h","yaw","pitch","eye_away","looking_at_camera"],eye_rows)

    # 接回音軌
    ff=None
    try:
        import imageio_ffmpeg; ff=imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff=shutil.which("ffmpeg")
    if ff:
        cmd=[ff,"-y","-i",tmp_mp4,"-ss",f"{start}","-i",args.video,"-t",f"{seg_dur}",
             "-map","0:v:0","-map","1:a:0?","-c:v","copy","-c:a","aac","-shortest",final_mp4]
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if r.returncode==0 and os.path.exists(final_mp4):
            os.remove(tmp_mp4)
        else:
            os.replace(tmp_mp4,final_mp4); print("(音軌接回失敗,輸出為無聲版)")
    else:
        os.replace(tmp_mp4,final_mp4); print("(找不到 ffmpeg,輸出為無聲版;裝 imageio-ffmpeg 可接回聲音)")

    # 說話分析(呼叫已除錯好的 speech_step4.py)
    if not args.no_speech:
        sp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"speech_step4.py")
        if os.path.exists(sp):
            print("\n說話分析中(呼叫 speech_step4.py,語言自動偵測)...")
            c=[sys.executable,sp,args.video,"-o",f"{prefix}_speech"]
            if args.start: c+=["--start",args.start]
            if args.end: c+=["--end",args.end]
            subprocess.run(c)
        else:
            print("\n(找不到 speech_step4.py,略過說話分析。放到同資料夾即可自動執行。)")

    # 摘要
    tot=sum(emo_count.values()) or 1
    print("\n"+"="*56)
    print("完成!")
    print(f"成品影片      : {final_mp4}(情緒+眼神+時間條,含聲音)")
    print(f"分析幀數      : {processed}(stride={args.stride})")
    print("情緒分布      : " + ", ".join(f"{EMO_ZH.get(e,e)} {emo_count[e]/tot*100:.0f}%"
          for e in EMOTIONS if emo_count[e]))
    if look_cnt[1]:
        print(f"眼神接觸      : {look_cnt[0]/look_cnt[1]*100:.1f}%(可分析臉 {look_cnt[1]})")
    print(f"CSV           : {prefix}_detections/emotions/eyecontact.csv")
    print("="*56)

if __name__ == "__main__":
    main()