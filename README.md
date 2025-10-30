# Lane Line Detection (車道線辨識)

## 動機 / 目的
- 降低車道線「一閃一閃」與誤抓隔壁車道
- 面對單線道→雙線道轉換時仍能保持穩定中心與寬度
- 產出可直接展示的 `lane_out.mp4` 影片

## 方法：程式
- 主要檔案：`mid.py`（OpenCV + Hough Transform）
- 執行：
  ```bash
  pip install -r requirements.txt
  python lane_main.py
