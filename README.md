# OTVideo
# 在其他電腦上執行本專案

治療師 POV 分析管線的安裝與執行說明。

## 一、要帶走哪些檔案

**必要(程式)**
```
tracking_pov_mediapipe_tasks.py   第一步:偵測 + 追蹤
fer_step2.py                      第二步:情緒辨識
eyecontact_step3.py               第三步:眼神接觸
requirements.txt                  相依套件(版本已鎖定)
check_env.py                      環境自我檢查
安裝說明.md                        本檔
```

**選用**
```
tracking_pov_anyvideo.py          第一步的離線 Haar 版(不需 mediapipe)
generate_pov_clip.py              合成測試影片產生器
```

**不需要帶(會自動重新產生/下載)**
- 模型檔(`*.tflite`、`*.onnx`、`*.task`)— 第一次執行自動下載
  （若目的地電腦網路受限,才需要手動複製過去,見第五節）
- 影片與輸出(`*.mp4`、`*.csv`)

---

## 二、安裝步驟

### 1. 安裝 Python
需 **Python 3.9 以上**(實測 3.13 可用)。
從 python.org 下載安裝時,**務必勾選「Add python.exe to PATH」**。裝完關掉終端機再重開。

確認:
```powershell
py --version
```

### 2. 安裝套件
把檔案放到同一個資料夾,`cd` 進去後:
```powershell
py -m pip install -r requirements.txt
```

### 3. 檢查環境
```powershell
py check_env.py
```
全部顯示 `[OK]` 才繼續。若有 `[FAIL]`,腳本會直接印出該執行的修復指令。

---

## 三、執行

三步依序執行(第二、三步都讀第一步產出的 CSV):

```powershell
# 第一步:偵測 + 追蹤
py tracking_pov_mediapipe_tasks.py video.mp4 -o out --start 1:00 --end 1:30

# 第二步:情緒辨識
py fer_step2.py video.mp4 out_detections.csv -o out_emo

# 第三步:眼神接觸
py eyecontact_step3.py video.mp4 out_detections.csv -o out_eye
```

第一次執行各步時會自動下載對應模型(需連外網),之後不會重複下載。

---

## 四、已知相依地雷(重要)

這些是實作過程中實際踩到的,換電腦時很可能再遇到:

| 問題 | 症狀 | 解法 |
|---|---|---|
| opencv 5.0.0 是殘缺 build | `module 'cv2' has no attribute 'CascadeClassifier'` | 移除後改裝 `opencv-python-headless==4.11.0.86` |
| mediapipe 新版移除舊 API | `module 'mediapipe' has no attribute 'solutions'` | 本專案已改用新版 Tasks API,直接裝最新版即可(勿降版) |
| Python 3.13 太新 | 舊版 mediapipe 找不到安裝檔 | 用最新版 mediapipe(≥0.10.30) |
| PowerShell 找不到指令 | `pip` / `yt-dlp` 無法辨識 | 一律改用 `py -m pip`、`py -m yt_dlp`(底線) |

`check_env.py` 會自動偵測上述前三項並給出修復指令。

---

## 五、網路受限的電腦

若目的地電腦無法連外網下載模型,把這三個檔案一起複製過去,放在與腳本相同的資料夾即可:

```
blaze_face_short_range.tflite    約 0.2 MB   第一步
emotion-ferplus-8.onnx           約 35 MB    第二步
face_landmarker.task             約 3.8 MB   第三步
```

腳本偵測到檔案已存在就不會重新下載。若下載到的檔案異常小,代表下載失敗,刪掉重跑即可(腳本會檢查大小並報錯)。

---

## 六、建議:用虛擬環境(選用)

若該電腦還有其他 Python 專案,建議隔離環境避免版本互相干擾:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```
之後每次使用前都要先執行 `.\venv\Scripts\Activate.ps1`。

若 PowerShell 擋住指令碼執行,先執行一次:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 七、長期建議:用 Git 管理

論文專案建議把程式放上 Git(GitHub 私有 repo),換電腦時 `git clone` 即可,也能保留修改歷史。

`.gitignore` 建議排除以下(檔案大、可重新產生):
```
*.mp4
*.csv
*.onnx
*.task
*.tflite
venv/
__pycache__/
```