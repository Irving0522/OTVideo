"""
clip_video.py  —  把影片裁成指定時間區間(保留聲音)
------------------------------------------------------
用途:把完整影片裁成分析區段(例如 1:00–1:30),裁好的片段從 0 秒開始,
方便丟進影片同步 demo——demo 會自動判斷這是「剪好的片段」並對齊。

不需另外安裝系統 ffmpeg:優先使用 pip 套件 imageio-ffmpeg 內建的 ffmpeg;
若系統已有 ffmpeg 也會自動採用。

安裝(擇一,若尚未有 ffmpeg):
    py -m pip install imageio-ffmpeg

用法:
    py clip_video.py <影片> --start T --end T [-o 輸出檔]
    時間 T 可用秒數或時間碼:90 / 1:30 / 01:02:03

範例:
    py clip_video.py interview.mp4 --start 1:00 --end 1:30
    → 產生 interview_clip_60-90.mp4(從 0 秒開始的 30 秒片段)
"""
import sys, os, argparse, subprocess, shutil

def parse_time(s):
    p = str(s).split(":")
    if len(p) == 1: return float(p[0])
    if len(p) == 2: return int(p[0])*60 + float(p[1])
    if len(p) == 3: return int(p[0])*3600 + int(p[1])*60 + float(p[2])
    raise ValueError(f"時間格式看不懂: {s}")

def find_ffmpeg():
    """優先用 imageio-ffmpeg 內建的 ffmpeg;否則找系統 PATH 上的 ffmpeg。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", required=True, help="起始時間(秒 或 MM:SS)")
    ap.add_argument("--end", required=True, help="結束時間(秒 或 MM:SS)")
    ap.add_argument("-o", "--output", help="輸出檔名(預設自動命名)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"找不到影片:{args.video}")

    start = parse_time(args.start)
    end = parse_time(args.end)
    dur = end - start
    if dur <= 0:
        sys.exit("結束時間必須大於起始時間。")

    ff = find_ffmpeg()
    if not ff:
        sys.exit("找不到 ffmpeg。請先安裝:  py -m pip install imageio-ffmpeg")

    if args.output:
        out = args.output
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out = f"{base}_clip_{int(start)}-{int(end)}.mp4"

    # -ss 放在 -i 前:快速定位;搭配重新編碼 → 精準且保留聲音。
    cmd = [ff, "-y", "-ss", f"{start}", "-i", args.video, "-t", f"{dur}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-movflags", "+faststart", out]

    print(f"使用 ffmpeg:{ff}")
    print(f"裁切 {args.start} → {args.end}(共 {dur:.1f}s)...")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        sys.exit(f"ffmpeg 執行失敗(代碼 {r.returncode})。")

    if os.path.exists(out):
        mb = os.path.getsize(out) / 1e6
        print(f"完成:{out}  ({mb:.1f} MB,{dur:.1f}s)")
        print("這個片段從 0 秒開始,丟進影片同步 demo 時會自動對齊(偏移=起始秒數)。")
    else:
        sys.exit("輸出檔未產生,請檢查上面的訊息。")

if __name__ == "__main__":
    main()