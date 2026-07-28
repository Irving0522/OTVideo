"""
speech_step4.py  —  POV 第四步子步驟:說話分析(轉錄 + 信心 + 語速)
================================================================================
以 faster-whisper 轉錄語音,輸出帶時間戳的片段,並計算數個「清晰度的替代指標」。

【重要:關於「說話清晰度」的科學誠實聲明】
真正臨床意義的說話清晰度(intelligibility,即「這個人講的話有多少比例能被聽懂」)
目前沒有可信的現成自動化工具。本腳本輸出的是「替代指標(proxy)」,不是清晰度分數:

  1. ASR 信心(avg_logprob → 機率):模型對自己轉錄結果的把握程度
  2. 逐詞信心:每個詞的機率,低信心詞比例可粗略反映發音不清
  3. 語速:字/詞 每秒
  4. 停頓:片段之間的間隔

【已知侷限(務必寫入論文限制)】
  - Whisper 以「成人語音」訓練,對兒童語音、構音異常語音辨識率明顯較差。
    因此「信心低」不必然等於「講不清楚」,也可能只是模型沒聽過這類語音。
    此侷限與第二步 FER 的成人偏誤屬同一類問題,同為本研究的動機。
  - 未做語者分離(diarization):治療師 POV 的麥克風會同時收到治療師與兒童的
    聲音,本腳本無法區分。輸出保留 speaker 欄位(目前為 unknown)以便日後接上。
  - 輔助線索:本腳本計算每段的 RMS 音量。治療師 POV 的麥克風在治療師頭上,
    治療師的聲音通常明顯較大、兒童較小,故音量可作為「誰在說話」的粗略線索
    (僅為啟發式,非正式語者分離)。

【時間軸對齊】
輸出時間戳為「影片絕對秒數」,與第一~三步一致,故可直接與影像模態融合。

安裝(本機):
    py -m pip install faster-whisper
    (不需另外安裝 ffmpeg:faster-whisper 透過 PyAV 解碼,已內建 ffmpeg 函式庫)
    第一次執行會自動下載 Whisper 模型。

用法:
    py speech_step4.py <影片或音檔> [-o 前綴] [--model small] [--lang zh]
        [--start T] [--end T] [--low-conf 0.6]

範例:
    py speech_step4.py interview.mp4 -o interview_speech --model small --lang en
    py speech_step4.py interview.mp4 -o interview_speech --start 1:00 --end 1:30
"""
import csv, os, math, argparse, re
import numpy as np

SR = 16000          # Whisper 固定取樣率
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

def parse_time(s):
    if s is None: return None
    p = str(s).split(":")
    if len(p) == 1: return float(p[0])
    if len(p) == 2: return int(p[0])*60 + float(p[1])
    if len(p) == 3: return int(p[0])*3600 + int(p[1])*60 + float(p[2])
    raise ValueError(f"時間格式看不懂: {s}")

def count_units(text):
    """回傳 (中文字數, 拉丁詞數)。中英混雜時兩者都算。"""
    n_cjk = len(CJK.findall(text))
    n_word = len(re.findall(r"[A-Za-z']+", text))
    return n_cjk, n_word

def rms_dbfs(samples):
    """音量(dBFS)。用來當『說話者遠近』的粗略線索。"""
    if samples.size == 0: return float("-inf")
    r = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(r, 1e-10))

def detect_pauses_acoustic(audio, sr, min_pause=0.25, frame=0.02, hop=0.01,
                           drop_db=25.0):
    """由『音訊訊號本身』偵測靜音停頓。

    為何不用 ASR 時間戳:Whisper 的逐詞時間戳是連續填滿的(前一詞的結束
    直接接下一詞的開始),並未把靜音編碼進去,因此無論用片段間隔或詞間隔
    都量不到真正的停頓。真正的停頓必須從聲學能量判定。

    方法(與 Praat 的靜音偵測同一套邏輯):
      1. 切成短時窗,算每窗的 RMS 音量(dB)
      2. 以第 95 百分位當作穩健的「語音音量」估計
      3. 低於「語音音量 - drop_db」者視為靜音窗
      4. 連續靜音達 min_pause 以上者,計為一次停頓
      5. 只看首尾語音之間,忽略錄音前後的空白

    回傳 (pauses, voiced_start, voiced_end),時間單位為秒(相對於傳入的音訊)。
    """
    fl, hl = int(frame*sr), int(hop*sr)
    if len(audio) < fl:
        return [], None, None
    nf = 1 + (len(audio) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(nf)[:, None]
    rms = np.sqrt(np.mean(audio[idx].astype(np.float64)**2, axis=1))
    db = 20*np.log10(np.maximum(rms, 1e-10))
    level = np.percentile(db, 95)
    silent = db < (level - drop_db)
    voiced = np.where(~silent)[0]
    if len(voiced) < 2:
        return [], None, None
    lo, hi = int(voiced[0]), int(voiced[-1])
    pauses, i = [], lo
    while i <= hi:
        if silent[i]:
            j = i
            while j <= hi and silent[j]:
                j += 1
            dur = (j - i) * hl / sr
            if dur >= min_pause:
                pauses.append({"start": i*hl/sr, "end": j*hl/sr, "dur": dur})
            i = j
        else:
            i += 1
    return pauses, lo*hl/sr, (hi*hl + fl)/sr

def pause_stats(pauses, span):
    """由停頓列表與說話時距算出統計。

    另區分語速與構音速率(流暢度評估的標準區分):
      speech rate       = 單位數 / 總說話時距(含停頓)
      articulation rate = 單位數 / 發聲時間(扣掉停頓)
    兩者差距大,代表話語被停頓切碎(流暢度問題的常見表現)。
    """
    if span is None or span <= 0:
        return None
    pause_time = sum(p["dur"] for p in pauses)
    durs = sorted(p["dur"] for p in pauses)
    return {
        "n": len(pauses), "span": span,
        "pause_time": pause_time,
        "pause_ratio": pause_time/span,
        "artic_time": max(1e-6, span - pause_time),
        "mean": sum(durs)/len(durs) if durs else 0,
        "median": durs[len(durs)//2] if durs else 0,
        "max": durs[-1] if durs else 0,
        "per_min": len(pauses)/(span/60) if span else 0,
        "n_short": sum(1 for d in durs if d < 1.0),
        "n_long": sum(1 for d in durs if d >= 1.0),
        "longest": max(pauses, key=lambda p: p["dur"]) if pauses else None,
    }

def summarize(rows, total_span, low_conf_thr):
    """由片段列表算出摘要統計。"""
    if not rows:
        return None
    speech = sum(r["duration"] for r in rows)
    confs = [r["confidence"] for r in rows]
    # 以片段長度加權的平均信心(長片段權重較高,較有代表性)
    wconf = sum(r["confidence"]*r["duration"] for r in rows) / speech if speech else 0
    rates_c = [r["rate_chars_per_sec"] for r in rows if r["rate_chars_per_sec"] > 0]
    rates_w = [r["rate_words_per_sec"] for r in rows if r["rate_words_per_sec"] > 0]
    low = [r for r in rows if r["confidence"] < low_conf_thr]
    return {
        "n_seg": len(rows), "speech": speech, "span": total_span,
        "speech_ratio": speech/total_span if total_span else 0,
        "conf_mean": sum(confs)/len(confs), "conf_weighted": wconf,
        "conf_min": min(confs), "conf_max": max(confs),
        "n_low": len(low), "low_dur": sum(r["duration"] for r in low),
        "rate_c": sum(rates_c)/len(rates_c) if rates_c else 0,
        "rate_w": sum(rates_w)/len(rates_w) if rates_w else 0,
        "tot_chars": sum(r["n_chars"] for r in rows),
        "tot_words": sum(r["n_words"] for r in rows),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media", help="影片或音檔")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default="small",
                    help="Whisper 模型:tiny/base/small/medium/large-v3(預設 small)")
    ap.add_argument("--lang", default=None,
                    help="語言代碼如 zh / en(不給則自動偵測)")
    ap.add_argument("--start", help="起始時間(秒 或 MM:SS)")
    ap.add_argument("--end", help="結束時間(秒 或 MM:SS)")
    ap.add_argument("--low-conf", type=float, default=0.6,
                    help="信心低於此值視為『低信心片段』(預設 0.6)")
    ap.add_argument("--pause-threshold", type=float, default=0.25,
                    help="靜音多久算一次停頓,單位秒(預設 0.25,語音研究常用值)")
    ap.add_argument("--silence-drop-db", type=float, default=25.0,
                    help="比語音音量低多少 dB 視為靜音(預設 25,同 Praat 慣例)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8",
                    help="cpu 建議 int8;有 GPU 可用 float16")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    prefix = args.output or os.path.splitext(os.path.basename(args.media))[0]
    seg_csv, word_csv = f"{prefix}_segments.csv", f"{prefix}_words.csv"
    pause_csv = f"{prefix}_pauses.csv"

    # ---------- 解碼音訊(PyAV,不需 ffmpeg 執行檔)----------
    print(f"解碼音訊:{args.media}")
    audio = decode_audio(args.media, sampling_rate=SR)
    full_dur = len(audio) / SR
    s = parse_time(args.start) or 0.0
    e = parse_time(args.end) or full_dur
    s, e = max(0.0, s), min(full_dur, e)
    if e <= s:
        raise SystemExit("結束時間必須大於起始時間。")
    clip = audio[int(s*SR):int(e*SR)]
    print(f"音訊長度 {full_dur:.1f}s,分析區間 [{s:.1f}s → {e:.1f}s]")

    # ---------- 轉錄 ----------
    print(f"載入 Whisper 模型 '{args.model}'(第一次會自動下載)...")
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    print("轉錄中...")
    segments, info = model.transcribe(
        clip, language=args.lang, word_timestamps=True,
        vad_filter=True)                      # VAD 過濾靜音,減少幻聽
    print(f"偵測語言:{info.language}(信心 {info.language_probability:.2f})")

    # ---------- 逐段計算指標 ----------
    rows, all_words = [], []
    prev_end = None
    for seg in segments:
        st, en = s + seg.start, s + seg.end          # 加回偏移 → 影片絕對秒數
        dur = max(1e-6, en - st)
        text = (seg.text or "").strip()
        n_cjk, n_word = count_units(text)
        conf = math.exp(seg.avg_logprob)             # logprob → 平均每 token 機率
        gap = 0.0 if prev_end is None else max(0.0, st - prev_end)
        # 該段在音訊陣列中的音量
        seg_samples = clip[int(seg.start*SR):int(seg.end*SR)]
        dbfs = rms_dbfs(seg_samples)

        # 逐詞信心(同時收集結構化資料供停頓分析)
        wprobs = []
        for w in (seg.words or []):
            wp = float(getattr(w, "probability", 0.0))
            wprobs.append(wp)
            all_words.append({"start": s + w.start, "end": s + w.end,
                              "word": (w.word or "").strip(), "prob": wp})
        low_word_ratio = (sum(1 for p in wprobs if p < args.low_conf) / len(wprobs)
                          if wprobs else 0.0)

        rows.append({
            "start": st, "end": en, "duration": dur, "text": text,
            "confidence": conf, "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "n_chars": n_cjk, "n_words": n_word,
            "rate_chars_per_sec": n_cjk/dur, "rate_words_per_sec": n_word/dur,
            "gap_before": gap, "rms_dbfs": dbfs,
            "n_word_tokens": len(wprobs), "low_word_ratio": low_word_ratio,
        })
        prev_end = en

    if not rows:
        print("\n未偵測到語音片段。可能是這段沒有人說話,或音軌有問題。")
        return

    # ---------- 輸出 ----------
    with open(seg_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["start_seconds", "end_seconds", "duration", "speaker",
                    "text", "confidence", "avg_logprob", "no_speech_prob",
                    "n_chars", "n_words", "rate_chars_per_sec",
                    "rate_words_per_sec", "gap_before", "rms_dbfs",
                    "n_word_tokens", "low_word_ratio"])
        for r in rows:
            w.writerow([f"{r['start']:.3f}", f"{r['end']:.3f}",
                        f"{r['duration']:.3f}", "unknown", r["text"],
                        f"{r['confidence']:.4f}", f"{r['avg_logprob']:.4f}",
                        f"{r['no_speech_prob']:.4f}", r["n_chars"], r["n_words"],
                        f"{r['rate_chars_per_sec']:.2f}",
                        f"{r['rate_words_per_sec']:.2f}",
                        f"{r['gap_before']:.3f}", f"{r['rms_dbfs']:.1f}",
                        r["n_word_tokens"], f"{r['low_word_ratio']:.3f}"])

    with open(word_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # 註:word_gap 為 Whisper 時間戳的詞間間隔,因其時間戳為連續填滿,
        #     此欄多接近 0,不可用來判定停頓(停頓見 *_pauses.csv)。
        w.writerow(["start_seconds", "end_seconds", "word", "probability",
                    "word_gap"])
        prev = None
        for wd in sorted(all_words, key=lambda x: x["start"]):
            gap = 0.0 if prev is None else max(0.0, wd["start"] - prev["end"])
            w.writerow([f"{wd['start']:.3f}", f"{wd['end']:.3f}", wd["word"],
                        f"{wd['prob']:.4f}", f"{gap:.3f}"])
            prev = wd

    # ---------- 報表 ----------
    st = summarize(rows, e - s, args.low_conf)
    print("\n" + "=" * 62)
    print("說話分析結果(注意:以下皆為清晰度的『替代指標』,非清晰度分數)")
    print("=" * 62)
    print(f"分析區間      : {s:.1f}s → {e:.1f}s(共 {st['span']:.1f}s)")
    print(f"語音片段      : {st['n_seg']} 段,總長 {st['speech']:.1f}s"
          f"(佔 {st['speech_ratio']*100:.1f}%)")
    print("\n--- ASR 信心(替代指標,非清晰度)---")
    print(f"  平均信心      : {st['conf_mean']:.3f}"
          f"(依長度加權 {st['conf_weighted']:.3f})")
    print(f"  範圍          : {st['conf_min']:.3f} ~ {st['conf_max']:.3f}")
    print(f"  低信心片段    : {st['n_low']} 段(< {args.low_conf}),"
          f"共 {st['low_dur']:.1f}s")
    raw_pauses, vs, ve = detect_pauses_acoustic(
        clip, SR, min_pause=args.pause_threshold, drop_db=args.silence_drop_db)
    # 轉為影片絕對秒數
    pauses = [{"start": s+p["start"], "end": s+p["end"], "dur": p["dur"]}
              for p in raw_pauses]
    pa = pause_stats(pauses, (ve - vs) if vs is not None else None)

    print("\n--- 語速 / 構音速率 ---")
    if pa:
        unit_c, unit_w = st["tot_chars"], st["tot_words"]
        if unit_c:
            print(f"  中文 語速      : {unit_c/pa['span']:.2f} 字/秒(含停頓)")
            print(f"       構音速率  : {unit_c/pa['artic_time']:.2f} 字/秒(扣掉停頓)")
        if unit_w:
            print(f"  英文 語速      : {unit_w/pa['span']:.2f} 詞/秒(含停頓)")
            print(f"       構音速率  : {unit_w/pa['artic_time']:.2f} 詞/秒(扣掉停頓)")
        print("  (兩者差距大 = 話語被停頓切碎,流暢度問題的常見表現)")
    else:
        if st["rate_c"]: print(f"  中文          : {st['rate_c']:.2f} 字/秒")
        if st["rate_w"]: print(f"  英文          : {st['rate_w']:.2f} 詞/秒")

    print(f"\n--- 停頓(由音訊靜音偵測,門檻 {args.pause_threshold}s / "
          f"-{args.silence_drop_db:.0f}dB)---")
    if not pa:
        print("  音訊過短或未偵測到語音,無法計算停頓。")
    elif pa["n"] == 0:
        print(f"  說話時距 {pa['span']:.1f}s 內未偵測到 >= "
              f"{args.pause_threshold}s 的靜音(語流連貫)。")
    else:
        print(f"  次數          : {pa['n']} 次"
              f"(短 {pa['n_short']} / 長(>=1s) {pa['n_long']})")
        print(f"  停頓頻率      : {pa['per_min']:.1f} 次/分鐘")
        print(f"  停頓時長      : 平均 {pa['mean']:.2f}s,中位數 {pa['median']:.2f}s,"
              f"最長 {pa['max']:.2f}s")
        print(f"  停頓佔比      : {pa['pause_time']:.1f}s / {pa['span']:.1f}s "
              f"= {pa['pause_ratio']*100:.1f}%")
        if pa["longest"]:
            g = pa["longest"]
            print(f"  最長停頓位置  : {g['start']:.2f}s → {g['end']:.2f}s")
        print(f"  (完整清單見 {pause_csv})")

    with open(pause_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["start_seconds", "end_seconds", "duration", "is_long"])
        for p in pauses:
            w.writerow([f"{p['start']:.3f}", f"{p['end']:.3f}",
                        f"{p['dur']:.3f}", 1 if p["dur"] >= 1.0 else 0])
    print("\n--- 音量(語者遠近的粗略線索,非正式語者分離)---")
    dbs = sorted(r["rms_dbfs"] for r in rows)
    if dbs:
        print(f"  範圍 {dbs[0]:.1f} ~ {dbs[-1]:.1f} dBFS,"
              f"中位數 {dbs[len(dbs)//2]:.1f} dBFS")
        print("  (治療師 POV:麥克風在治療師頭上,治療師通常明顯較大聲)")

    print("\n--- 最低信心的 3 段(最可能是發音不清或模型不確定)---")
    for r in sorted(rows, key=lambda x: x["confidence"])[:3]:
        t = r["text"][:40] + ("..." if len(r["text"]) > 40 else "")
        print(f"  [{r['start']:7.2f}s] 信心 {r['confidence']:.3f}  {t}")

    print(f"\n輸出: {seg_csv}")
    print(f"輸出: {word_csv}")
    print(f"輸出: {pause_csv}")
    print("\n提醒:Whisper 以成人語音訓練,對兒童/構音異常語音辨識較差。")
    print("      信心低不必然代表講不清楚,亦可能是模型不熟悉該類語音。")

if __name__ == "__main__":
    main()