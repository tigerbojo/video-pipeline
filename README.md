# AI 影片自動後製工作流

**一鍵將戶外拍攝素材轉為完整 YouTube 影片** — AI 旁白、語音複製、自動字幕、AI 配樂，全部在本地 GPU 上完成。

---

## 功能總覽

### 兩種工作模式

| 模式 | 用途 | 流程 |
|------|------|------|
| **AI 旁白後製** | 風景/戶外素材 | 粗剪 → AI 看圖寫旁白 → 語音合成 → 字幕 → 配樂 → 合成 |
| **原音上字幕** | 訪談/會議/Vlog | Whisper 辨識 → AI 潤詞 → 說話者辨識 → 燒入字幕 |

### 七步驟 Pipeline（AI 旁白模式）

```
0. 去人聲（選用）    — demucs AI 分離 / FFmpeg 中央聲道消除
1. 自動粗剪         — auto-editor 靜音移除
2. AI 旁白稿        — Ollama Gemma 4 vision 看圖寫稿（長度自動匹配影片）
   ↓ 審閱編輯 ↓
3. AI 配音          — edge-tts / GPT-SoVITS 語音複製
4. 字幕生成         — edge-tts WordBoundary 精準對時 / Whisper ASR
5. AI 配樂          — ACE-Step 1.5 本地生成 / 靜音佔位
6. 合成輸出         — FFmpeg 動態 ducking 混音 + 字幕燒入
```

### 原音上字幕模式

```
上傳影片 → Whisper 語音辨識 → OpenCC 簡→繁 → AI 潤詞（去贅詞）
         → 說話者辨識（SpeechBrain）→ 審閱編輯 → 燒入字幕
```

---

## 快速開始

### 系統需求

- **Python** 3.10+
- **FFmpeg**（必備）
- **GPU**（建議）：NVIDIA GPU + CUDA，大幅加速 Whisper 和 GPT-SoVITS
- **Ollama**（AI 旁白需要）：本地或區網均可

### 安裝

```bash
# 1. Clone
git clone https://github.com/tigerbojo/video-pipeline.git
cd video-pipeline

# 2. 一鍵安裝（Mac/Linux）
bash setup.sh

# 或手動安裝（Windows）
pip install -r requirements.txt

# 3. 啟動
python app.py

# 4. 開啟瀏覽器
# http://localhost:7860
```

### 選裝工具

| 工具 | 用途 | 安裝方式 |
|------|------|---------|
| auto-editor | 智慧靜音移除 | `pip install auto-editor` |
| demucs | AI 去人聲 | `pip install demucs` |
| CUDA PyTorch | GPU 加速 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |

---

## 詳細功能說明

### AI 旁白生成

- 使用 **Ollama Gemma 4 e4b** 本地 vision 模型
- 自動從影片擷取關鍵幀（場景切換偵測，非固定間隔）
- 旁白字數自動匹配影片長度（中文語速 ~3.5 字/秒）
- 支援自訂旁白風格（Discovery 紀錄片 / 輕鬆 Vlog / 詩意文藝）
- 區網 Ollama 支援：UI 可設定遠端 Ollama 位址

### 語音合成 / 聲音複製

| 引擎 | 說明 | 需要 |
|------|------|------|
| **edge-tts** | 微軟免費語音，5 種中文角色 | 無（免費） |
| **GPT-SoVITS** | 聲音複製，用你的聲音配旁白 | GPT-SoVITS 伺服器 |

**GPT-SoVITS 聲音複製流程：**
1. 錄製 3-10 秒聲音樣本（系統自動裁切+降噪+正規化）
2. Whisper 自動辨識樣本內容作為 prompt_text
3. 長文本自動切割（300 字/段）→ 逐段合成 → 拼接
4. 品質驗證：輸出異常短時自動降級到 edge-tts

**GPT-SoVITS 伺服器安裝：**
```bash
# 下載 Windows 整合包（~8GB，含 Python + CUDA + 模型）
# https://huggingface.co/lj1995/GPT-SoVITS-windows-package

# 解壓後啟動 API
cd GPT-SoVITS-v2pro-xxx
runtime/python.exe -I api_v2.py
# 伺服器跑在 http://127.0.0.1:9880
```

### 字幕生成

- **AI 旁白模式**：edge-tts WordBoundary 逐字對時（毫秒級精準）
- **原音上字幕模式**：Whisper large-v3 (GPU) / medium (CPU)
- **OpenCC s2twp**：自動簡體→台灣繁體（含台灣用語）
- **AI 潤詞**：Ollama Gemma 4 自動刪除贅詞（嗯、啊、那個、對啊...）
- **說話者辨識**：SpeechBrain ECAPA-TDNN（免 token，跨平台）

### Whisper 模型選擇

| 模型 | 精準度 | GPU 速度 | CPU 速度 |
|------|:---:|:---:|:---:|
| large-v3 | 最佳 | 快 | 慢 |
| medium | 很好 | 很快 | 中等 |
| base | 普通 | 極快 | 快 |

首次使用會自動下載模型（~1-3GB），之後快取不再下載。

### 混音與合成

- **動態 ducking**：旁白播放時自動降低原始音軌和背景音樂（FFmpeg sidechaincompress）
- **無音軌影片支援**：自動偵測，不會因缺少音軌而崩潰
- **輸出時長**：跟隨原始影片長度，不會被 BGM 拉長

---

## 專案架構

```
video-pipeline/
├── app.py                          # Gradio Web UI（雙模式、分步驟 Tab）
├── setup.sh                        # Mac/Linux 一鍵安裝
├── requirements.txt                # Python 依賴
├── pipeline/
│   ├── __init__.py                 # Pipeline orchestrator + RunTracker
│   ├── base.py                     # Step 基礎框架（required/optional flag）
│   ├── context.py                  # PipelineContext dataclass（型別安全）
│   ├── batch.py                    # 批次佇列（persistent JSON + atomic write）
│   ├── run_metadata.py             # 結構化 run 記錄
│   ├── vocal_remove.py             # Step 0: 去人聲
│   ├── rough_cut.py                # Step 1: 自動粗剪
│   ├── narration.py                # Step 2: AI 旁白稿
│   ├── voiceover.py                # Step 3: AI 配音
│   ├── subtitle.py                 # Step 4: 字幕生成
│   ├── music.py                    # Step 5: AI 配樂
│   ├── merge.py                    # Step 6: 合成輸出
│   ├── transcribe.py               # 逐字稿（辨識+潤詞+說話者）
│   └── engines/
│       ├── whisper_pool.py         # Whisper 模型快取（singleton）
│       ├── gpt_sovits.py           # GPT-SoVITS HTTP client
│       ├── ace_step.py             # ACE-Step 1.5 音樂生成
│       ├── text_splitter.py        # 中文句子切割
│       └── audio_utils.py          # FFmpeg 音頻工具
└── workspace/                      # 工作暫存（gitignored）
```

---

## 跨機器部署

本專案設計為可在多台機器間協作：

| 機器 | 角色 | 設定 |
|------|------|------|
| **A 電腦**（有 GPU） | 主工作站：跑 pipeline + Ollama + GPT-SoVITS | 預設設定即可 |
| **Mac Mini M2** | 輕量工作站 | Ollama 位址改為 A 電腦 IP（如 `http://192.168.0.198:11434`） |

Mac Mini 安裝：
```bash
git clone https://github.com/tigerbojo/video-pipeline.git
cd video-pipeline && bash setup.sh
# 啟動後在 UI 把 Ollama 位址改為 A 電腦的 IP
```

---

## 技術特色

- **全本地運行**：不依賴雲端 API（Ollama + Whisper + GPT-SoVITS + ACE-Step）
- **Windows cp950 安全**：全域 subprocess UTF-8 patch，不會因中文路徑崩潰
- **Whisper 模型快取**：singleton pool，第二次辨識起省 30-60 秒
- **場景切換取樣**：FFmpeg scene detection 擷取關鍵幀，旁白更貼合畫面
- **聲音樣本前處理**：自動去靜音、正規化音量、降噪、highpass/lowpass
- **動態 ducking**：sidechaincompress 旁白播放時自動壓低背景音
- **Pipeline 韌性**：required/optional 步驟區分，選用步驟失敗不中斷流程
- **Codex 交叉審查**：經 4 輪 OpenAI Codex 獨立審查，修正 20+ 個問題

---

## 授權

MIT License

---

## 致謝

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 語音複製
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 語音辨識
- [edge-tts](https://github.com/rany2/edge-tts) — 免費語音合成
- [auto-editor](https://github.com/WyattBlue/auto-editor) — 智慧靜音移除
- [ACE-Step](https://github.com/ace-step/ACE-Step-1.5) — AI 音樂生成
- [SpeechBrain](https://github.com/speechbrain/speechbrain) — 說話者辨識
- [OpenCC](https://github.com/yichen0831/opencc-python) — 簡繁轉換
- [Gradio](https://gradio.app/) — Web UI 框架
- [Ollama](https://ollama.com/) — 本地 LLM 推理
