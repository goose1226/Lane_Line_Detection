# Lane Line Detection (車道線辨識)

以 OpenCV + Hough Transform 實作「穩定、不晃動」的車道線偵測與可視化，支援輸出 MP4 影片  
Robust **lane line detection** using OpenCV and Hough Transform, with stabilized rendering and MP4 export.

---

## 🎯 動機 (Motivation)
- **安全與輔助駕駛需求**：車道線（lane lines）是 ADAS（Advanced Driver Assistance Systems，先進駕駛輔助系統）與自動駕駛的基本環境線索，穩定偵測能支援偏移警示、路徑規劃與巡航控制  
- **實務痛點**：許多教學範例在彎道、亮度變化或「單線道 → 雙線道」時，容易一閃一閃或誤抓隔壁車道，本專案針對這些情境做 **防外線拉走** 與 **缺線容忍** 的優化  
- **離線可展示**：純 OpenCV 不依賴雲端或大型模型，在一般筆電/桌機即可即時運行，並輸出 MP4 方便課堂/比賽展示  
- **教學與快速原型**：完整呈現傳統電腦視覺流程：前處理 → 邊緣（Canny）→ 線段（HoughLinesP）→ 幾何約束（ROI/寬度限制）→ 穩定化（EMA）→ 視覺化疊圖

## 🎯 目的 (Objectives)
- **穩定不閃爍**：以 **EMA（Exponential Moving Average，指數移動平均）** 平滑斜率/截距，並設 `MISS_MAX` 容忍短暫缺線  
- **不被外側車道帶走**：以搜尋窗 `WINDOW_PX` 只接受「上一幀左右邊界附近」的候選線，並用寬度上下限（`HALF_MIN/HALF_MAX`）與每幀成長上限（`MAX_HALF_GROWTH`）避免單線道→雙線道時整個面被拉寬  
- **即時與可移植**：在 640×400 解析度達到可視即時（依硬體調整），純 CPU 即可跑  
- **結果可用**：輸出具備上淡下濃漸層的可讀性疊圖與多編碼回退（`avc1/mp4v/XVID/MJPG`），確保能產生可播放的影片檔

---

## 🧠 方法：程式 (Method: Code)
**Pipeline**
1. 影像前處理（dilate → Gaussian blur → erode）  
2. 邊緣偵測（Canny，`L2gradient=True`）  
3. ROI 裁切（上底收窄，降低撿到隔壁車道）  
4. 線段偵測（`cv2.HoughLinesP`）＋斜率篩選（避免太平/太陡）  
5. **左右線挑選**：以「底部交點靠近中心的左右邊界」為準  
6. **時間平滑**：對左右線的斜率 `s`、截距 `b` 做 EMA  
7. **寬度防護**：限制半寬變化與全域上下限，避免突然變兩線道  
8. 視覺化（多邊形上色：上淡下濃漸層 + 邊界 + 中線）  
9. 影片輸出（MP4 / AVI，多 codec 回退）

**Quick Start**
```bash
# 安裝依賴
pip install -r requirements.txt

# 放入輸入影片（檔名固定）
# └─ LaneVideo.mp4  (你的原始道路影片)

# 執行
python mid.py

# 產出
# └─ lane_out.mp4   (偵測結果疊圖影片；若 mp4 不支援會自動改用 AVI)
